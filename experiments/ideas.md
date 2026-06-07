# Idea backlog — candidate improvements

Prioritized hypotheses to try. **Research the actual paper before implementing** — these are
pointers, not verified claims. Move an idea to `lab_notebook.md` once tried. Bias toward ideas that
(a) help *convergence* (we only train ~60 min → ~22 epochs, early-schedule) and (b) are
latency-neutral at inference. Keep `verify on COCO later` in mind: prefer changes whose mechanism
should generalize beyond VisDrone's small/dense objects; VisDrone-specific ones are flagged.

Each Tier-1 idea below carries: paper, the exact single change + files, why it helps convergence
under the cap, latency risk, complexity cost, expected effect, and COCO-transfer.

---

## Tier 1 — fast-convergence / latency-neutral (best fit for the walltime cap)

### 1. Scale contrastive denoising (`num_denoising`)  ← top pick: simplest, lowest-risk
- **Paper / source:** Contrastive DeNoising (CDN), DINO (Zhang et al., ICLR 2023, arXiv:2203.03605);
  inherited by RT-DETR / D-FINE. CDN feeds noised GT boxes as extra decoder queries that must
  denoise back to GT — pure auxiliary supervision, **stripped at inference**.
- **Exact change (one key):** raise `base_cfg["DFINETransformer"]["num_denoising"]` in
  `src/d_fine/configs.py:19` from `100` → try `300` (sweep 200/300 in one comparison is fine since
  it's a single knob). Optionally co-set via a config override instead, but the value lives in
  `configs.py`. **One change.**
- **Why it should help here:** denoising groups are `num_group = num_denoising // max_gt_num`
  (`src/d_fine/arch/utils.py:380`). VisDrone images are object-dense, so `max_gt_num` is large
  (often >100) → with `num_denoising=100` the floor `num_group=1` kicks in: **we're running the
  minimum possible denoising supervision.** Raising it restores multiple noised-GT groups per step
  → far denser, stable positive supervision exactly when O2O matching is sparse early in training —
  the dominant lever under a 22-epoch cap.
- **Latency risk:** **none at inference.** Denoising is fully gated behind `self.training`
  (`dfine_decoder.py:971`) and split off before eval/export (`:1007-1014`) → the exported graph is
  byte-identical. Latency-neutral by construction.
- **Complexity cost:** ~1 line. Lowest of any idea here. Honors the simplicity rule trivially.
- **Trade-off to watch:** more denoising tokens = slightly heavier decoder attention *during
  training* → marginally slower steps → marginally fewer epochs in the 60-min cap. Denoising tokens
  are cheap vs the encoder, so expect a small hit; if steps/epoch drops noticeably, that offsets the
  gain — note it in the ledger.
- **Expected effect:** faster, more stable convergence; modest mAP_50_95 gain, f1 neutral-to-up.
- **COCO transfer:** ✅ general DETR mechanism; the under-supply is worse on dense VisDrone but CDN
  scaling helps COCO too (this is how DINO/RT-DETR are trained).

### 2. DEIM Dense O2O (heavy mosaic)
- **Paper / source:** DEIM, CVPR 2025 (arXiv:2412.04234). Dense O2O = use Mosaic/MixUp to pack more
  objects per training image → many more one-to-one positive matches per step → denser supervision.
  Reduces DETR training time ~50% on RT-DETR/D-FINE in the paper.
- **Exact change:** raise mosaic supervision in `configs/research_visdrone.yaml` overrides — e.g.
  `train.mosaic_augs.mosaic_prob` 0.8→1.0 and/or add MixUp if the aug pipeline supports it; the
  effective single lever is "more dense-mosaic exposure." Touches the aug config (and `src/dl/train.py`
  only if MixUp wiring is missing). **Keep it to one knob** (start with `mosaic_prob`→1.0); do not
  also touch the loss.
- **Why it should help here:** more positives per image directly attacks O2O sparsity, the main
  convergence bottleneck under the cap.
- **Latency risk:** none — training-time aug only; inference graph unchanged.
- **Complexity cost:** low if `mosaic_prob` alone; medium if MixUp needs adding (then weigh simplicity
  rule). Mind CLAUDE.md gotcha #6 (mosaic discouraged for *segment* — we're on detect, OK) and GUIDE
  §8 (mosaic-close schedule never reaches its end under the cap — consistent across runs, fine).
- **Expected effect:** convergence speed-up; mAP_50_95 up if denser supervision outweighs the harder
  augmented distribution in only ~22 epochs.
- **COCO transfer:** ✅ DEIM is validated on COCO with D-FINE.
- **MAL note:** the MAL loss-half was tried alone (2026-06-07) and **rejected as a near-tie** — it's
  designed to manage the low-quality matches Dense O2O *introduces*, so it likely only pays off
  *with* Dense O2O. After Dense O2O lands, consider re-testing **MAL + Dense O2O** together (loss
  code is on branch `exp/mal`). That pair is two changes — sequence Dense O2O first.

### 3. Muon optimizer for 2D matrices (hybrid with AdamW)
- **Paper / source:** Muon (Jordan et al., 2024; "MomentUm Orthogonalized by Newton–Schulz").
  Orthogonalizes the momentum update for 2D weight matrices → faster convergence per step
  (~35% on nanoGPT speedruns) — the exact lever that matters under a 60-min cap.
- **Exact change:** in `build_optimizer` (`src/d_fine/dfine.py:124`) route 2D hidden matrices of the
  encoder/decoder (attention/MLP linears) to Muon, keep norms, biases, embeddings, the detection
  head, and the backbone on AdamW (a hybrid optimizer). The repo already builds 4 custom param
  groups there, so the split point exists. Touches `src/dl/train.py`/`build_optimizer` only (allowed
  under GUIDE rule 4). **One change** (the optimizer); tune Muon LR separately, do not also touch the
  schedule.
- **Why it should help here:** the encoder/decoder transformer is full of 2D linears with
  high-condition-number, near-low-rank gradients — Muon's sweet spot; better per-step progress is
  worth more here than anywhere because we only get ~22 epochs.
- **Latency risk:** none — optimizer only; inference graph unchanged. (Also uses less optimizer
  memory than Adam: one momentum buffer vs two.)
- **Complexity cost:** **medium** — adds an optimizer dependency/implementation + a param-group split
  + a separate Muon LR to tune. Exploratory (evidence outside LLMs is still thin). If the gain is
  marginal, the simplicity rule says drop it.
- **Expected effect:** high-upside but uncertain; faster early convergence → higher mAP at the cap.
- **COCO transfer:** ✅ optimizer-level, architecture-agnostic.

### 4. Group-DETR one-to-many auxiliary query groups (training-only)
- **Paper / source:** Group DETR (Chen et al., 2022, arXiv:2207.13085). K parallel query groups,
  O2O assignment *per group* (so K positives/GT), self-attention isolated per group via mask. At
  inference only one group is used → **architecture/latency unchanged.**
- **Exact change:** in the decoder (`src/d_fine/arch/dfine_decoder.py`) replicate the query set into
  K groups during training, extend the self-attention mask to block cross-group attention (the CDN
  attn-mask machinery at `:971` is a template), and match each group one-to-one in the criterion.
  Drop extra groups when `not self.training`. **One conceptual change** (add o2m groups) but spans
  decoder + matcher wiring.
- **Why it should help here:** K× more positive supervision per image with no change to the O2O
  inference behavior — strong convergence accelerant under the cap.
- **Latency risk:** none at inference (extra groups dropped). Higher *training* compute/memory.
- **Complexity cost:** **high** — touches decoder attention masking + criterion; most code of any
  idea here. Given idea #1 (CDN scaling) delivers similar "more training-only positives" with ~1
  line, **prefer #1 first**; only reach for Group-DETR if #1's gains plateau and we want more.
  Weigh hard against the simplicity rule.
- **Expected effect:** convergence speed-up; mAP up.
- **COCO transfer:** ✅ general DETR mechanism, validated on COCO.

### 5. EMA decay tuned for short runs  ← cheap, but flag as walltime-specific
- **Paper / source:** standard EMA-of-weights practice; decay sets the averaging window.
- **Exact change:** `train.ema_momentum` (config `0.9998`) → try `0.999`. One config key (override in
  `configs/research_visdrone.yaml`). The EMA also has a warmup ramp `0.9998*(1-exp(-x/2000))`
  (`src/dl/train.py:66`) — leave that untouched; change only the asymptote.
- **Why it should help here:** `0.9998` averages over ~5000 iters — tuned for long COCO schedules.
  In a ~17k-iter run where weights are *still improving* at the cap, that window lags the live model;
  a faster EMA (`0.999`, ~1000-iter window) tracks the still-moving optimum better.
- **Latency risk:** none.
- **Complexity cost:** trivial (1 key).
- **Expected effect:** small mAP/f1 bump at the cap.
- **COCO transfer:** ⚠️ **walltime-specific.** This is tuned to the short 60-min run; a long COCO
  run wants the high decay back. Lowest transfer value here — treat as a campaign micro-opt, not a
  COCO-bound gain. Run it only as a cheap filler, and don't carry the value into a COCO run.

---

## Tier 2 — loss / assignment
- Varifocal / quality-focal refinements to the VFL classification target.
- IoU-aware or distribution-focal regression tweaks (D-FINE already uses FDR/FGL — look for
  orthogonal gains, not duplicates).
- Matcher cost reweighting for small/dense objects (`matcher.weight_dict` in `configs.py:42`) —
  ⚠️ **VisDrone-specific, likely non-transferring to COCO; flag for manual COCO check.**

## Tier 3 — architecture (watch the latency budget carefully)
- Encoder/neck feature-fusion tweaks (cheaper or stronger cross-scale fusion).
- Activation / normalization swaps in encoder/decoder.
- Query-selection method (`query_select_method` in `configs.py:25`) alternatives.

## Notes / constraints
- Init is **ImageNet backbone only** — backbone-heavy ideas start from pretrained HGNetv2; neck/head
  ideas start random. Account for that when reading results.
- Anything that adds inference params/FLOPs must clear the *larger* accuracy margin (see the
  decision function) AND justify its complexity (simplicity rule) to be worth the latency.
- Tier-1 ranking rationale: #1 (CDN scaling) is the simplest latency-neutral convergence lever and
  VisDrone is currently starved (groups collapse to 1) → do it first. #2 (Dense O2O) is the
  established DEIM backlog item and unlocks the MAL re-test. #3/#4 are bigger convergence bets with
  more code; #5 is a cheap but non-transferable filler.
