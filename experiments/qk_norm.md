# QK-norm — shelved stability fix (problem, implementation, why it's not in the repo)

**Status (2026-06-10): NOT in the trunk.** Full working code lives on branch `exp/qk-norm-lr`
(arch `f45f71f`, COCO-loader remap `a5445ab`); trained artifacts in
`experiments/runs/qknorm_full75/seed42` and `experiments/runs/qk-norm-lr/seed{42,123}`.
This doc is the recipe to re-add it if it's ever needed.

## 1. The problem it solves

DETR-family training can NaN-diverge on long runs / higher LRs (issue #64: stable for 225 epochs,
then loss → 2e36 and never recovers; `muon-lr`: NaN at epoch 16 at peak LR 0.01). Root cause chain:
nothing bounds the attention logits `q·k/√d` — weights drift up over training until one batch
overflows under fp16 AMP (65504 ceiling), then inf → NaN cascades through softmax/matcher.
(YOLO-class models don't have this failure: BN everywhere + bounded anchor-relative boxes.)

**QK-norm** (Dehghani et al., ViT-22B; Chameleon): LayerNorm Q and K *per head* right before the
dot product. Logits become bounded by construction, regardless of weight scale.

Evidence it works (VisDrone, `s`):
- `muon-lr` @ peak 0.01 → NaN at ep16. Same LR + qk-norm → **both seeds train clean**.
- Accuracy: neutral-to-positive. 2-seed screen +0.0015 mAP_50_95 (within margin);
  **full 75-ep COCO-init run: test mAP_50_95 0.2388 vs muon_full75 0.2359 (+0.0029)**,
  train-eval f1 0.5661 vs 0.5633, tracked ≥ muon at every epoch.
- Torch latency unchanged (even slightly faster than nn.MultiheadAttention).

## 2. Why it's NOT in the repo

**TensorRT mis-executes fully-trained qk-norm checkpoints at every precision** (verified
exhaustively 2026-06-10 on RTX 5070 Ti / sm_120, TRT 10.13–11.0):

- qknorm_full75: torch/ORT-fp32/ORT-fp16 all agree (f1 0.582-equiv; 928/929/926 dets on identical
  inputs) — but TRT fp32 = 0.552 (scores scatter ±0.5 vs torch), TRT fp16+GridSample-pin = 0.545.
  TRT 10.16/11.0: fp32 0.529, and 10.16 NaNs the fp16 build outright.
- It is a **weights-dependent TRT graph-compiler (Myelin) defect**: the screen checkpoint produces a
  *structurally identical* 877-node ONNX and compiles correctly (0.552 ≈ torch 0.554); only the
  trained constants differ. Every op is correct in isolation (standalone GridSample with captured
  real inputs: maxdiff 1e-5). Ruled out: fp16, TF32, opt levels, onnxsim, the SDPA export pattern
  (explicit-attention export), grid clamping, fp32 tails/pins, VERSION_COMPATIBLE.
- The muon (nn.MultiheadAttention) decoder never trips it — muon_full75 is at exact parity
  (0.585 @ 2.1 ms) under the same export.

Since the end deliverable is the fp16 TRT engine, qk-norm is undeployable on this stack → shelved.

**Conditions to revisit:** a new TRT release (or driver/ptxas change, or different target GPU)
verified on the *full-trained* checkpoint — re-bench `experiments/runs/qknorm_full75/seed42`
(rebuild engine, TRT-row f1 must ≈ torch 0.582 @ thr 0.55, ~2.1 ms); or a non-TRT deployment
target (ONNXRuntime/OpenVINO execute it correctly); or an NVIDIA bug fix (artifact set: two weight
sets on one graph flip correctness, ORT ground truth, version matrix).

## 3. How to implement (complete recipe)

Reference diff: `git diff main_exp exp/qk-norm-lr -- src/` shows all of this in place.

### 3.1 The attention module — `src/d_fine/arch/utils.py`

```python
class QKNormSelfAttention(nn.Module):
    """Self-attention with per-head QK LayerNorm — drop-in for nn.MultiheadAttention (batch_first)."""

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead, self.head_dim, self.dropout = nhead, d_model // nhead, dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, query, key, value, attn_mask=None):
        B, Lq, _ = query.shape
        q = self.q_norm(self.q_proj(query).view(B, Lq, self.nhead, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(key).view(B, key.shape[1], self.nhead, self.head_dim)).transpose(1, 2)
        v = self.v_proj(value).view(B, value.shape[1], self.nhead, self.head_dim).transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # nn.MultiheadAttention bool mask (True = block) -> additive -inf, the SDPA convention
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(attn_mask, float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, Lq, -1)), None
```

The returned `(out, None)` tuple matches nn.MultiheadAttention's call signature, so call sites
don't change. The bool-mask conversion is required for the decoder's denoising mask.

### 3.2 Threading (5 sites)

- `arch/dfine_decoder.py` `TransformerDecoderLayer.__init__`: `qk_norm=False` param;
  `self.self_attn = QKNormSelfAttention(d_model, n_head, dropout) if qk_norm else nn.MultiheadAttention(...)`.
  Pass through `DFINETransformer.__init__` into both `decoder_layer` and `decoder_layer_wide`.
- `arch/hybrid_encoder.py` `TransformerEncoderLayer.__init__`: same pattern; thread via
  `HybridEncoder.__init__`.
- `src/d_fine/dfine.py` `build_model(..., qk_norm: bool = False)`:
  `model_cfg["HybridEncoder"]["qk_norm"] = qk_norm; model_cfg["DFINETransformer"]["qk_norm"] = qk_norm`.
- `config.yaml`: `train.qk_norm: False` (default off — existing users unaffected).
- `src/dl/train.py` + `src/dl/export.py` `prepare_model`: pass `qk_norm=cfg.train.qk_norm` /
  `cfg.train.get("qk_norm", False)` into `build_model`.

### 3.3 Checkpoint self-detection — `src/infer/torch_model.py`

Checkpoints self-describe via the `q_norm` keys, so inference wrappers rebuild the right arch:

```python
state_dict = torch.load(self.model_path, weights_only=True, map_location="cpu")
qk_norm = any("q_norm" in k for k in state_dict)
self.model = build_model(..., qk_norm=qk_norm)
```

### 3.4 Warm COCO init — `src/d_fine/utils.py` (sha `a5445ab`)

Stock checkpoints store self-attn as fused `in_proj_weight` ([3d, d] = q,k,v stacked); the split
projections won't key-match, silently leaving 24 attention tensors random. Add to
`load_tuning_state`, right after `extract_pretrained_state_dict`:

```python
def remap_fused_qkv(model_state, pretrain_state):
    """Split nn.MultiheadAttention fused in_proj into q/k/v_proj when the model uses the
    qk-norm attention, so pretrained self-attn weights load warm instead of silently dropping."""
    for k in list(pretrain_state):
        if not (k.endswith("self_attn.in_proj_weight") or k.endswith("self_attn.in_proj_bias")):
            continue
        base, kind = k.rsplit("in_proj_", 1)
        if f"{base}q_proj.{kind}" not in model_state:
            continue
        for name, part in zip(("q", "k", "v"), pretrain_state[k].chunk(3, dim=0)):
            pretrain_state[f"{base}{name}_proj.{kind}"] = part
```

Verified: 801/826 tensors load from `dfine_s_coco.pt` (only the new q/k LayerNorms + class heads
stay fresh), and the full75 run tracked muon from epoch 1 — **no COCO re-pretraining needed**.

### 3.5 Tests

`tests/unit/test_qk_norm.py` on the branch: shapes/grad, bool-mask-blocks-like-MHA, and
remap slicing == torch's in_proj convention.

### 3.6 Validation protocol before trusting any deployment

Train ≥ a screen run, `make export && make bench`, and require **TRT-row f1 ≈ torch-row f1** on the
test set — a healthy torch model can still produce a broken engine (this is exactly how the TRT
defect was caught; the guard rule is EXPERIMENT_GUIDE §3).
