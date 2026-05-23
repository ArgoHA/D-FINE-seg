# D-FINE-seg — Deep Repository Audit

Date: 2026-05-22
Scope: `src/dl/`, `src/d_fine/`, `src/infer/` (training, model, loss, inference, eval).
Method: static read-through of the core pipeline. No code was changed (except the pre-existing
uncommitted edit to `src/dl/bench.py`, which is unrelated to this audit).

Severity legend: 🔴 high · 🟠 medium · 🟡 low

---

## 1. Bugs

### 🟠 B3 — `train()` crashes with `NameError` on an empty train loader

`src/dl/train.py:695`: after the batch loop,
`if (batch_idx + 1) % self.b_accum_steps != 0 ...` references `batch_idx`. If the dataloader
yielded nothing (empty split, or every item filtered to `None`), `batch_idx` was never bound and
this raises `NameError` instead of a clear message. Initialize `batch_idx = -1` before the loop.

### 🟠 B4 — `finally` block in `train.main` masks the real error and can crash on its own

`src/dl/train.py:780-797`: training exceptions are swallowed with `logger.error(e)` and the
process proceeds into the `finally` block, which unconditionally does
`torch.load(path_to_save / "model.pt")`. If the run crashed before the first checkpoint was
saved, this raises `FileNotFoundError` from inside `finally`, hiding the original failure. The
process can also exit with status 0 after a failed run, so CI / wrappers think it succeeded.

Fix: guard the `model.pt` existence check; re-raise (or exit non-zero) when training did not
complete; only run final eval when a checkpoint exists.

### 🟡 B5 — Box edge coordinates clamped one pixel short

`src/dl/utils.py:160-185` and `src/infer/torch_model.py:462-486` (`norm_xywh_to_abs_xyxy`,
`to_round=True` path): `x_max`/`y_max` are clamped to `width - 1` / `height - 1`. In the
xyxy convention `x_max`/`y_max` are exclusive and may legitimately equal `width`/`height`.
Every object touching the right/bottom edge loses ~1px. Eval impact is small (GT and preds are
both clamped through the same path), but YOLO labels written by `infer.py` and exported boxes
are systematically off-by-one. Clamp `x_max`/`y_max` to `width`/`height`.

### 🟡 B7 — `figure_input_type` can use an undefined variable

`src/dl/infer.py:18-32`: `data_type` is only assigned inside the `for` loop. A folder that
contains no recognized image/video extension (e.g. only hidden files or `.txt`s) falls through
and `logger.info(... data_type ...)` raises `NameError`. Give `data_type` a default and emit a
clear error when nothing usable is found.

---

## 2. Model accuracy improvements

### 🟡 A3 — `_focal_loss_mask` is implemented but unused

`src/d_fine/dfine_criterion.py:310-335` defines an adaptive-alpha focal mask loss that
`loss_masks` never calls (it uses `_cropped_bce_loss` + `_cropped_dice_loss`). Either wire it in
behind a config flag or delete it — currently it is misleading dead code. Worth A/B testing
focal-BCE vs. cropped-BCE for datasets with very small instances.

### 🟡 A4 — `loss_masks` will hard-crash on mixed detect/segment annotations

`src/d_fine/dfine_criterion.py:549-553`: if a `segment` run ever sees an image whose matched
queries exist but whose `targets["masks"]` is empty (e.g. a detection-only label file mixed into
a seg dataset), `_prepare_target_masks` skips that image while `pred_sel` still includes its
queries → shape mismatch → `AssertionError` kills training. Currently masked by the dataset
always producing one (possibly empty) mask per box, but it is fragile. Filter `pred_sel` by the
same per-image validity instead of asserting.

---

## 3. Speed improvements

### 🟡 S3 — Validation mask IoU is computed at full original resolution

`src/dl/validator.py:_pairwise_mask_iou` (283-293) flattens masks to `[N, H*W]` float32 and
does `pmf @ gmf.T`. For high-res datasets (e.g. 4K) with up to 300 predictions/image this is a
very large matmul per image and dominates eval time. IoU is scale-robust — resizing both pred
and GT masks to a fixed small grid (e.g. 256×256) before IoU gives near-identical numbers at a
fraction of the cost. Also a memory win (see H2).

### 🟡 S4 — `_test_pred` warmup uses a fixed odd-sized random image

`src/infer/torch_model.py:81-85`: harmless, but the warmup forward runs at `1100×1000` which,
under `keep_ratio`/`rect`, triggers a differently-shaped graph than real inputs. For CUDA-graph
/ TRT-style caching this means the first *real* call still pays a recompile/reshape. Warm up at
the actual `input_size` instead.

---

## 4. Hardware / memory improvements

### 🟠 H1 — Eval accumulates per-image masks for the whole split before computing metrics

`src/dl/train.py:get_preds_and_gt` (391-449) collects `all_preds` / `all_gt` for the entire
val/test set in RAM (RLE-encoded, which helps), but `Validator` then **deep-copies** them
(`validator.py:58`, `96-97`) and `_prepare_masks_for_torchmetrics` decodes RLE back to dense
per batch. Peak host RAM ≈ (RLE store) + (one `mask_batch_size` worth of dense masks) +
(deep copy). On large/high-res datasets this is the dominant RAM cost of a run.
Mitigations: stream metric computation per-batch without the full-dataset deepcopy; lower
`mask_batch_size`; store/evaluate masks at a fixed reduced resolution (ties into S3/H2).

### 🟠 H2 — Mask IoU matmul materializes large float32 tensors

`src/dl/validator.py:287-292`: `pmf` is `[Np, H*W]` in float32. For a 1920×1080 image and
`Np≈300` that is `300 · 2.07M · 4 B ≈ 2.5 GB` for `pmf` alone, plus `gmf` and the result.
Evaluating mask IoU on a fixed small grid (256×256) cuts this ~30–60× and removes a real OOM
risk on consumer GPUs/CPUs. Lowest-risk high-impact memory fix.

### 🟡 H3 — `auto_batch_size` probes with a single repeated sample and fixed shape

`src/dl/utils.py:1452-1461`: the probe builds the batch via `sample_img.repeat(bs,...)` and
`sample_targets * bs` at a fixed `img_size`. Real training applies multi-scale augmentation
(`train_collate_fn`, `dataset.py:856-864`) which enlarges inputs by up to `+64px`. The probe
therefore underestimates peak VRAM and the chosen batch size can OOM mid-epoch on a
multi-scale-up batch. Probe at `img_size + max multiscale offset`, or apply a safety margin to
`target_fraction`.

### 🟡 H4 — `persistent_workers=True` keeps a post-validation-bloated parent forked

`src/dl/dataset.py:741-746` already notes train workers persist. Because workers are forked
*once* (at first epoch) from the parent, and the parent's RSS grows after each validation pass
(metrics, plots, torchmetrics), the persisted workers carry copy-on-write pages from a heavier
parent than necessary. The existing `rebuild_train_loader` path (used after `close_mosaic` /
`ignore_background`) re-forks; consider also that workers never benefit from the parent's later
memory being freed. Low priority — flagged for awareness, not urgent.

---

## 5. Quick-reference summary

| ID | Sev | Area | One-line |
|----|-----|------|----------|
| B3 | 🟠 | bug | `NameError` on empty train loader (`batch_idx`) |
| B4 | 🟠 | bug | `finally` masks training errors, can crash on missing `model.pt` |
| B5 | 🟡 | bug | xyxy `x_max`/`y_max` clamped 1px short |
| B7 | 🟡 | bug | `figure_input_type` undefined `data_type` |
| A3 | 🟡 | accuracy | unused `_focal_loss_mask` (wire in or remove) |
| A4 | 🟡 | accuracy | `loss_masks` asserts/crashes on mixed annotations |
| S3 | 🟡 | speed | eval mask IoU at full resolution |
| S4 | 🟡 | speed | warmup uses wrong input shape |
| H1 | 🟠 | memory | full-split mask accumulation + deepcopy in eval |
| H2 | 🟠 | memory | mask IoU matmul materializes multi-GB float32 |
| H3 | 🟡 | memory | `auto_batch_size` ignores multi-scale upscale → OOM risk |
| H4 | 🟡 | memory | persistent workers forked from bloated parent |

### Suggested priority order
1. **H2 / S3** — evaluate mask IoU on a fixed small grid; removes OOM risk and speeds eval.
2. **B3 / B4** — robustness around empty loaders and failed runs.
3. Remaining 🟡 items as cleanup.

---

## Resolved

- **B2 / S1** (2026-05-23) — DDP all-reduced grads on every micro-step instead
  of only the final one in each accumulation window — `N`× the necessary wire
  traffic with `b_accum_steps = N`.
  Fixed: `train.py` now wraps non-boundary backwards in `self.model.no_sync()`
  when DDP is active and `b_accum_steps > 1`. A `had_synced_backward_in_window`
  flag plus a local `force_grad_sync()` helper cover the edge case where the
  boundary micro-step is skipped for non-finite loss — without it, earlier
  no_sync'd grads would never sync and ranks would diverge silently. Same
  fallback applied to the end-of-epoch leftover-grads path. No correctness
  change in the happy path; backward-path comms drop by ~`b_accum_steps`×.
- **S2** (2026-05-23) — `wandb.watch` logged gradient/param histograms every step.
  Fixed: removed the `wandb.watch` call in `train.py`. Metric logging (`wandb.init`,
  `wandb.log`, `wandb_logger`) is unaffected.
- **B6** (2026-05-23) — inner-loop variable shadowed the outer in `bench.test_model`.
  Fixed: inner loop now binds `target` (`src/dl/bench.py`), so iteration is correct
  regardless of bench batch size.
- **B8** (2026-05-23) — `run_images` dereferenced `None` on unreadable images.
  Fixed: `infer.py` now checks `cv2.imread` result, logs a warning, and skips the file.
- **B9** (2026-05-23) — mosaic invalid-polygon path kept stale boxes; dead `valid_indices`.
  Fixed: `dataset.py` now drops boxes whose polygon is fully clipped away and applies
  `valid_indices` to filter `mosaic_targets` (det-only rows with no polygon are still kept).
- **B1 / A1** (2026-05-22) — `build_loss` shared-`losses`-list mutation / doubled mask loss.
  Fixed: `merge_configs` now `deepcopy`s base + values so model sizes don't share list objects;
  `build_loss` `deepcopy`s `models[model_name]` and guards against duplicate `"masks"`.
- **A2** (2026-05-22) — mask metrics evaluated vs. low-res GT rasterization.
  Fixed: `CustomDataset.__getitem__` now also returns the untransformed original-resolution
  polygons for val/test (`polys_out`, aligned with boxes via `surviving_indices`); collated
  into the target dict under `"polys"`. `train.gt_postprocess` and `bench.test_model`
  rasterize GT masks directly from those polygons at original resolution instead of the
  network-size rasterize → upsample round-trip. Training-time GT still rasterizes at network
  resolution. Since `polys` and `masks` are produced by the same `surviving_indices`
  selection, an empty `polys` always coincides with empty masks (background image / detection
  task), so eval simply emits an empty mask tensor — no separate fallback path needed.
- **B3** to be ignored as it doesn't achieve anything. Empty dataloader will fail training anyways.
</content>
</invoke>
