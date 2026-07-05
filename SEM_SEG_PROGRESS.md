# SEM_SEG_PROGRESS.md — sem_seg implementation status & findings

Companion to [SEM_SEG_PLAN.md](SEM_SEG_PLAN.md). Phase 1 (core training pipeline) is **done**;
phases 2–3 (inference, export/bench) are **not started**. Read this before working on them.

## Done (Phase 1 — training pipeline)

| Piece | Where | Notes |
|---|---|---|
| `SemSegDecoder` | `src/d_fine/arch/dfine_decoder.py` (after `MaskDecoder`) | fuser (attr named `mask_decoder` for weight transfer) → neck (256→128→128) → Dropout2d → 1×1 classifier; logits at 1/4, bilinear ×4 to full res; train-only aux head on `feats[0]` |
| `SemSegCriterion` | `src/d_fine/sem_seg_criterion.py` | CE + multi-class soft Dice + 0.4×aux-CE; weights in `configs.py` `base_cfg["SemSegCriterion"]`; all-ignore batch → zero loss (graph kept for DDP) |
| task dispatch | `src/d_fine/dfine.py` | `build_model(..., task=...)` / `build_loss(..., task=, ignore_index=, class_weights=)` — kwargs default to old behavior, so export/infer call sites are untouched |
| `SemSegDataset` + `sem_seg_collate_fn` | `src/dl/dataset.py` | reads `masks/<stem>.png` grayscale; Compose-level `mask_interpolation=cv2.INTER_NEAREST`; Rotate/CoarseDropout/PadIfNeeded fill mask with `ignore_index`; optional `train.sem_seg.scale_jitter` (RandomScale→Pad→RandomCrop **after** resize); targets = `[{"sem_mask", "orig_size"}, ...]` per image |
| `SemSegValidator` | `src/dl/validator.py` | streaming [C,C] pixel confusion matrix, **original-resolution protocol** (pred argmax → NEAREST upsample → compare vs re-read GT PNG); mIoU over GT-present classes; per-class IoU in `extended_metrics.csv`; row-normalized confusion-matrix png |
| train wiring | `src/dl/train.py` | `evaluate()` short-circuits to `evaluate_sem_seg()`; `decision_metrics` forced to `["mIoU"]`; DDP path all-reduces the confusion matrix (written, not yet tested on 2 GPUs) |
| eval visualization | `src/dl/utils.py` `visualize_sem_seg` | GT \| pred side-by-side overlays (first 20 val images) → `output/eval_preds/`; aug debug overlays → `output/debug_images/` |
| config | `config.yaml` | `task: sem_seg`, `train.sem_seg.{ignore_index, class_weights, scale_jitter}`, 23 drone-dataset classes, `pretrained_model_path: pretrained/dfine_seg_${model_name}_coco.pt` |
| tests | `tests/unit/test_sem_seg.py`, `tests/integration/test_cpu_forward.py` | decoder shapes/aux, criterion ignore_index + all-ignore-zero, validator math + ignore pixels, NEAREST id preservation, rotate→255 fill; nano sem_seg CPU smoke. `make test-fast`: 98 passed |

Verified end-to-end on the Semantic Drone Dataset (400 imgs, 6000×4000, 23 classes,
339/61 split): weight transfer from `dfine_seg_s_coco.pt` leaves exactly the 13 new
neck/classifier/aux keys random (`unmatched: []`), mIoU improves monotonically, overlays and
per-class CSVs look right.

## Training results (dev dataset)

- Smoke (s, 640², bs 8, 3 epochs): val mIoU 0.103 → 0.124 → 0.128, pixel_acc 0.71.
- Baseline `semseg_s_base_2026-07-05` (s, 640², auto-bs 14, 75 epochs, defaults: plain
  resize, no scale_jitter, muon+adan, bf16 AMP, EMA; 0.61 h on RTX 5070 Ti):
  **val mIoU 0.6437, pixel_acc 0.8947**. Per-class IoU exactly as plan §1.1 predicted:
  large regions strong (pool 0.97, roof 0.94, car 0.92, water 0.91, paved-area/grass 0.88),
  thin structures weak (fence-pole 0.28, door 0.07, window 0.44) — the levers there are
  input res 640→768 first, then `out_stride: 2`. "unlabeled" 0.10 is a heterogeneous
  catch-all, not a model failure. wandb: `argo-cve/aerial_sem_seg/runs/69z4ait1`.
- First ablation to run next (plan §0): `train.sem_seg.scale_jitter: [0.75, 1.5]` vs this
  0.6437 baseline.

## Findings / gotchas for future phases

1. **`auto_batch_size` probe collate**: it builds its own `DataLoader` with a hardcoded
   `train_collate_fn`; had to swap in `sem_seg_collate_fn` for sem_seg. Anything else that
   builds ad-hoc loaders must do the same (grep for `train_collate_fn`).
2. **`Trainer.__init__` failures are silent in nohup runs** — `main()`'s try/except only wraps
   `trainer.train()`, so init-time exceptions (like the collate bug above) go to stderr only.
   Keep the launch log.
3. **Compose-level `mask_interpolation=cv2.INTER_NEAREST`** (albumentations 2.0.8) overrides
   every geometric transform in one place — don't set per-transform values, they'd be ignored.
4. **`keep_ratio: True` is rejected** for sem_seg (`SemSegDataset.__init__` raises):
   `LetterboxRect` has no ignore_index mask fill. If letterbox support is ever needed, add
   `apply_to_mask` with a configurable fill first, and undo padding in eval/postprocess.
5. **Eval reads original-res GT PNGs each epoch** (~60 × 24MP ≈ few seconds). Fine here;
   for large val sets consider caching downscaled GT only if the protocol changes (it's
   pinned to original res deliberately — resized-GT mIoU reads ~1–2 pts higher).
6. **EMA model is what's evaluated/saved** (repo default). With tiny datasets/few iters per
   epoch the EMA lags early — first-epoch mIoU looks worse than the raw model would.
7. **Muon works as-is for sem_seg**: `_is_muon_param` matches only encoder AIFI matrices
   (SemSegDecoder has no attn/linear1/linear2 names), rest goes to the aux optimizer. No
   scheduler changes needed.
8. **`out_stride: 2` knob (plan §1.1) is not implemented** — only 1/4 output exists. Same for
   OHEM CE. Both are explicitly later knobs.
9. **Model output contract** (already consumed by train/eval): train mode
   `{"sem_seg_logits": (B,C,H,W), "sem_seg_logits_aux": (B,C,H,W)}`; eval mode logits only.
   Phase 2/3 must key on `sem_seg_logits` and argmax; the planned wrapper output key is
   `out["sem_seg"]` (plan §4).

## Not touched (phases 2–3)

- `src/infer/*` (all backends), `src/dl/infer.py` — no sem_seg path; `Torch_model` ctor is
  still `enable_mask_head`-bool based (plan: make it task-aware).
- `src/dl/export.py` — would currently fuse `DFINEPostProcessor` and fail on missing
  `pred_logits`; needs the skip + fused-argmax int32 output + parity swap (plan §6).
- `src/dl/bench.py`, `test_batching.py`, `check_errors.py`, `ov_int8.py` / `trt_int8` — untouched.
- `src/etl/split.py` `ignore_negatives` still checks `labels/` (plan open question: should
  check `masks/` for sem_seg). Harmless with `ignore_negatives: False`.
- Gradio `demo/` — detection/instance only.
