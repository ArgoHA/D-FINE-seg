# SEM_SEG_PLAN.md — add semantic segmentation as a third task

Goal: add `task: sem_seg` alongside `detect` / `segment` so the framework can produce a
dense per-pixel class map (e.g. a drone traversability / landing-safety mask). Instance
segmentation (`task: segment`, YOLO polygons) stays exactly as-is.

## 0. Locked design decisions

- **Task flag**: `task: sem_seg` (third value; `detect` / `segment` unchanged).
- **Head**: reuse encoder + `MaskDecoder` fuser, bypass the whole `DFINETransformer`
  query/matcher/denoising path. No queries, no NMS, no boxes. On top of the fuser: a small
  **seg neck** (conv3×3+GN+ReLU ×2, 256→128→128) + `Dropout2d(0.1)` + 1×1 classifier —
  without queries the entire classification burden is per-pixel; a bare 1×1 conv is too thin.
  Plus a **train-only aux head** on the stride-8 PAN feature (deep supervision). See §1.
- **Pretrained init**: `dfine_seg_<size>_coco.pt` (instance-seg COCO weights; on HF and
  already matched by `ensure_pretrained`'s filename regex) — transfers backbone + encoder +
  the **entire `MaskDecoder` fuser** (`decoder.mask_decoder.*` keys). Keep the attr name
  `mask_decoder` in `SemSegDecoder` so keys line up; neck/classifier/aux train from scratch
  (`strict=False` skips them). Plain detect weights also load (fuser from scratch).
- **Loss**: Cross-Entropy + Dice (multi-class) + 0.4 × aux-CE.
- **Data**: single-channel PNG masks, **1 pixel = 1 class** (pixel int value = class id).
  YOLO polygons are untouched and remain the instance-seg path.
- **`ignore_index: 255` honored everywhere**: GT pixels, aug-introduced padding (rotate
  corners, letterbox bars, crop padding, CoarseDropout holes), loss, and metric. See §2.
- **Geometry**: plain resize to input size by default (repo default `keep_ratio: False`;
  resize > letterbox found empirically on D-FINE detect). Optional **scale-jitter + crop**
  aug knob (§2) — applied *after* the deployment-identical resize so train/test GSD match;
  it fills mosaic's scale-augmentation role for sem_seg. First accuracy ablation to run.
- **Output stride**: logits at 1/4, bilinear ×4 to full res (default). `out_stride: 2` knob
  adds a light ×2 refinement stage for a later accuracy-vs-latency A/B (§1.1). Start at 4.
- **Class map**: reuse `label_to_name` as-is (every pixel, incl. background/unsafe, is a
  class in it, e.g. `{0: unsafe, 1: safe}`). No separate sem_seg class list; `num_classes =
  len(label_to_name)`.
- **Decision metric**: mIoU only (config keeps the `decision_metrics` mechanism; a comment
  notes sem_seg uses IoU only). No box-F1 / mAP for this task. Protocol pinned in §5:
  confusion matrix accumulated at **original image resolution**.
- **Inference disambiguation** (Q5): distinct output key. Instance masks stay under
  `out["masks"]` = `[N, H, W]`; semantic output goes under a **new key `out["sem_seg"]`** =
  `[H, W]` label map. The wrapper is also constructed task-aware, so callers never have to
  guess from shape. See §4.

## 0.1 Dev/test dataset

All training sessions and tests during development use the **Semantic Drone Dataset**
(TU Graz ICG) at `/home/argo/Desktop/Projects/aerial_sem_seg/data/dataset`. Format is already a
drop-in for §2: `images/<stem>.jpg` (3-ch, 6000×4000) + `masks/<stem>.png` (single-channel
uint8, **pixel value = class id**), 400 matched pairs. Class ids are contiguous 0-indexed
**0–22 → `num_classes = 23`** (`class_dict_seg.csv` lists a 24th "conflicting"=23 that never
appears in any mask; drop it). `label_to_name` = the 23 names from that CSV (paved-area,
dirt, grass, …, obstacle). GT is fully labeled (no 255 pixels), so `ignore_index=255` is
exercised only through the aug pad-fill path (rotate/letterbox/crop) — sufficient for dev.
Native 6000×4000 → plain `A.Resize` to input size doubles as a stress test of the resize
geometry. Domain-relevant: remaps cleanly to a `{0: unsafe, 1: safe}` landing mask later.

## 1. Architecture

The expensive part is built and pretrained: `MaskDecoder` (`src/d_fine/arch/dfine_decoder.py:336`)
fuses encoder PAN features into a `(B, 256, H/4, W/4)` map, and `dfine_seg_<size>_coco.pt`
ships trained weights for it. The AIFI transformer inside HybridEncoder plays the
global-context role (ASPP/PPM equivalent), so no extra context module is needed.

Add a standalone decoder that plugs into the existing `__inject__` "decoder" slot so
`DFINE.forward` (`dfine.py:39`) needs **no change** — it already calls
`self.decoder(x, targets, low_level_feat=...)`:

```python
# src/d_fine/arch/dfine_decoder.py (new, next to MaskDecoder)
def conv_gn_act(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch), nn.ReLU(inplace=True))

class SemSegDecoder(nn.Module):
    def __init__(self, num_classes, feat_channels, mask_dim=256, mask_low_level_ch=None,
                 neck_dim=128, dropout=0.1, aux=True):
        super().__init__()
        in_chs = list(feat_channels)
        if mask_low_level_ch is not None:          # nano: prepend backbone 1/8 feat
            in_chs = [mask_low_level_ch] + in_chs
        # attr name "mask_decoder" → dfine_seg_<size>_coco.pt fuser weights transfer
        self.mask_decoder = MaskDecoder(in_chs=in_chs, out_ch=mask_dim)
        self.neck = nn.Sequential(conv_gn_act(mask_dim, neck_dim),
                                  conv_gn_act(neck_dim, neck_dim))
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(neck_dim, num_classes, 1)
        # train-only deep supervision on the stride-8 PAN feature
        self.aux_head = nn.Sequential(
            conv_gn_act(feat_channels[0], neck_dim), nn.Dropout2d(dropout),
            nn.Conv2d(neck_dim, num_classes, 1)) if aux else None

    def forward(self, feats, targets=None, low_level_feat=None):
        mask_feats = list(feats) if low_level_feat is None else [low_level_feat] + list(feats)
        x = self.mask_decoder(mask_feats)                     # (B, 256, H/4, W/4)
        logits = self.classifier(self.dropout(self.neck(x)))  # (B, C, H/4, W/4)
        logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
        out = {"sem_seg_logits": logits}                      # (B, C, H, W)
        if self.training and self.aux_head is not None:
            aux = self.aux_head(feats[0])                     # feats[0] = finest PAN level
            out["sem_seg_logits_aux"] = F.interpolate(
                aux, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        return out
```

Notes:
- Aux head reads `feats[0]` (finest PAN level; stride 8 for s/m/l/x, 16 for nano) and
  interpolates by `size=`, not a fixed factor, so nano works unchanged. Dropped at export
  (`self.training` gate) — zero inference cost.
- Neck cost at 640 input (1/4 = 160×160): 256→128 conv ≈ 7.5 GMACs, 128→128 ≈ 3.8. Real but
  affordable next to the fuser's existing 256→256 up_conv (~15 GMACs). `neck_dim` is the
  latency knob if it bites on small models.

`build_model` (`dfine.py:61`) picks the decoder by task and keeps the nano low-level-feat
logic (`dfine.py:87-93`) — generalize its gate from `enable_mask_head` to
`enable_mask_head or task == "sem_seg"`.

### 1.1 Output stride: start at 1/4, keep 1/2 as a knob

Default `out_stride: 4` (above). `out_stride: 2` inserts one more stage after the neck:
bilinear ×2 → conv3×3 (neck_dim→64, GN, ReLU) → classifier at 1/2 → bilinear ×2. Cost at
640: ≈ 7.5 GMACs (a 256-ch stage at 1/2 would be ~60 GMACs — never do that).

Why start at 4: (a) the pretrained fuser covers exactly the 1/8→1/4 path — a 1/2 stage is
from-scratch either way; (b) full-res CE on bilinear-upsampled logits already recovers most
boundary sharpness; (c) literature: 1/8→1/4 carries the boundary gains, 1/4→1/2 is
diminishing returns (mostly thin structures). If boundaries disappoint, try **input res
640→768 first** (usually the better lever at equal latency), then `out_stride: 2`.

## 2. Data path

Masks live at `<train.data_path>/masks/<stem>.png` (single channel, uint8, value = class id;
`ignore_index` value, default 255, excluded from loss + metric). `make split`
reuses as-is (it lists `images/`; only touches `labels/` when `ignore_negatives`, which we
switch to check `masks/` for sem_seg).

New **lean `SemSegDataset`** in `src/dl/dataset.py` (separate from `CustomDataset`, reuses
module helpers `read_image_hwc` / `_read_image` conventions):

- Reads image + `masks/<stem>.png` (`cv2.IMREAD_GRAYSCALE`).
- Albumentations compose **without** `bbox_params`, with `mask=` target and
  **`mask_interpolation=cv2.INTER_NEAREST`** — critical: LINEAR corrupts integer class ids.
  (The instance path's `INTER_LINEAR` at `dataset.py:299` is fine for binary masks, wrong here.)
- **Every transform that introduces pixels fills the mask with `ignore_index`**, else pad
  pixels get class 0 and (with `{0: unsafe}`) train the model that borders are "unsafe"
  while inflating val mIoU:
  - `A.Rotate` (`dataset.py:281`): add `fill_mask=ignore_index` (image fill stays 114).
  - `A.CoarseDropout`: `fill_mask=ignore_index` — don't supervise classes under occluders.
  - `LetterboxRect` (only if `keep_ratio: True`): needs `apply_to_mask` with a configurable
    mask fill — verify it has one before supporting keep_ratio for sem_seg.
  - Scale-jitter's `PadIfNeeded` (below): `fill_mask=ignore_index`.
- **Geometry**: train default = plain `A.Resize` to input size (deployment-identical).
  Optional `train.sem_seg.scale_jitter: [lo, hi]` (e.g. `[0.75, 1.5]`) enables, after the
  resize: `A.RandomScale` → `A.PadIfNeeded(fill=114, fill_mask=ignore_index)` →
  `A.RandomCrop(input_size)`. Applied post-resize so train GSD matches the resized frames
  seen at inference. Val/test/bench: plain resize always.
- No mosaic for sem_seg (polygon/box-centric); scale-jitter+crop is its replacement.
- `__getitem__` returns `(image, sem_mask[H,W] long, image_path, orig_size)` — path kept so
  eval can re-read the original-res GT (§5).
- New `sem_seg_collate_fn`: stack images `[B,C,H,W]`, stack masks `[B,H,W]`, target dict
  `{"sem_mask": ..., "orig_size": ..., "image_path": ...}`.

`Loader.build_dataloaders` (`dataset.py:855`) dispatches to `SemSegDataset` +
`sem_seg_collate_fn` when `task == "sem_seg"`.

## 3. Loss

New small `SemSegCriterion` (own file `src/d_fine/sem_seg_criterion.py`, or a branch in
`build_loss` at `dfine.py:109`):

```python
loss_ce   = F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)
loss_dice = dice_loss(softmax(logits), one_hot(target), ignore_index)  # multi-class soft dice
loss_aux  = F.cross_entropy(aux_logits, target, ignore_index=ignore_index)  # train only
return {"loss_ce": w_ce * loss_ce, "loss_dice": w_dice * loss_dice, "loss_aux": 0.4 * loss_aux}
```

Weights in `configs.py` base_cfg (new `SemSegCriterion.weight_dict`: `loss_ce`, `loss_dice`,
`loss_aux`). Optional `class_weights` for imbalance (traversability is usually skewed) —
config-driven, default uniform. OHEM CE is a possible follow-up knob for harder imbalance —
not in v1. Train loop (`train.py:641`) already does `sum(loss_dict.values())`, so no change
there once `build_loss` returns the sem_seg criterion.

## 4. Inference & output contract (Q5 answer)

`Torch_model` (`src/infer/torch_model.py`) is constructed task-aware (replace the
`enable_mask_head` bool with a `task` string, or add `sem_seg=` flag). For sem_seg its
`_preds_postprocess` skips the entire topk/conf/NMS/box block and instead:

1. `logits = outputs["sem_seg_logits"]` → argmax over C → `[B, H, W]`.
2. Undo letterbox padding + resize to original size with **NEAREST** (a `process_sem_seg`
   helper mirroring `process_masks` at `torch_model.py:108`, but nearest + no per-box cleanup).
3. Return per-image dict `{"sem_seg": label_map_uint8_[H,W]}` — **no** `labels`/`boxes`/`scores`.

**Disambiguation rule (definitive):** downstream code keys on the dict field.
`"masks"` ⇒ instance masks `[N,H,W]`; `"sem_seg"` ⇒ dense label map `[H,W]`. Distinct keys,
self-documenting for the standalone wrappers users copy out of `/infer`, no shape guessing.

`src/dl/infer.py` gets a sem_seg branch in `run_images` / `run_videos`:
- Visualize = blend a per-class color palette over the image (new `Visualizer` path or a
  small `colorize(label_map, palette)` overlay).
- Save GT-style output = write the predicted label map PNG under `labels/` (or `masks/`),
  not YOLO txt. Skip crops/tracking (both are box-based).

## 5. Metrics / Validator

Add a lightweight **`SemSegValidator`** (own class; the existing `Validator` is box/instance
heavy). Accumulate a `[C, C]` pixel confusion matrix across the dataset (memory-cheap —
no storing dense masks), then:

- per-class IoU = `diag / (row + col - diag)`, **mIoU** = mean over present classes,
- pixel accuracy, optional per-class F1 (informational).

**Protocol (pinned): original resolution.** Upsample pred argmax to `orig_size` with NEAREST
and compare against the original-res GT PNG (re-read via the image path — a grayscale read
per val image is cheap). The batch-level resized GT is used only for the loss. One protocol
everywhere: per-epoch val, final eval, bench. GT pixels == `ignore_index` never enter the
matrix.

`train.py` wiring:
- `enable_mask_head`/task derivation (`train.py:127`) → add sem_seg branch.
- decision-metric swap block (`train.py:138-143`) → when sem_seg, set
  `self.decision_metrics = ["mIoU"]`.
- `get_preds_and_gt` / `evaluate` (`train.py:416-527`) → for sem_seg, collect argmax maps +
  GT maps and feed `SemSegValidator` (skip RLE/box postprocess). `gt_postprocess` /
  `preds_postprocess` get a sem_seg short-circuit returning label maps.
- `save_model` (`train.py:529`) already averages `decision_metrics` present in `metrics` — works
  once `metrics = {"mIoU": ...}`.

## 6. Export / bench

- `export.py` (`prepare_model` `:121`, `main` `:644`): for sem_seg, **skip** the
  `DFINEPostProcessor` fusion (there's no detection postprocess). `ExportWrapper` **fuses the
  argmax** into the graph and outputs the label map as **int32**, name `sem_seg` — int64
  outputs upset TRT, and `H×W` int32 vs `C×H×W` fp16 logits is a real output-bandwidth win
  for video-rate inference. Keep a debug/parity escape hatch that exports raw logits.
- `run_parity` (`:479`) compares detection scores — for sem_seg swap to per-pixel argmax
  agreement % (or logit cosine on the debug output) between torch and each backend. Keep
  warn gating.
- `bench.py`: sem_seg branch reports mIoU + latency (reuse `SemSegValidator`). Can land
  after the torch path works.

## 7. Config additions (`config.yaml`)

```yaml
task: detect   # detect | segment | sem_seg  (sem_seg = dense per-pixel classes; uses IoU/mIoU only)
train:
  # sem_seg: masks live in <data_path>/masks/<stem>.png, 1 channel, pixel value = class id
  sem_seg:
    ignore_index: 255          # pixel value excluded from loss + mIoU (also used as aug mask fill)
    class_weights: null        # optional per-class CE weights for imbalance
    scale_jitter: null         # e.g. [0.75, 1.5] → RandomScale+Pad+RandomCrop after resize
    out_stride: 4              # 4 (default) | 2 (adds light ×2 refinement stage, see plan §1.1)
  decision_metrics:            # sem_seg overrides this to [mIoU] at runtime
  - f1
  - mAP_50
  - iou
```

For sem_seg runs, point `train.pretrained_model_path` at
`pretrained/dfine_seg_${model_name}_coco.pt` (same manual switch as instance-seg runs today).
`label_to_name` doubles as the sem_seg class map (0-indexed, contiguous — includes the
"background/unsafe" class since every pixel gets a label).

## 8. Tests (`tests/`)

- Unit: `SemSegDecoder` forward shape `(B,C,H,W)` + aux only in train mode; `SemSegCriterion`
  returns finite CE+Dice+aux and respects `ignore_index` (an all-ignore target contributes
  zero loss); `SemSegValidator` mIoU on a hand-built confusion matrix + ignore pixels never
  counted; NEAREST mask aug preserves ids; pad-introducing augs fill mask with 255.
- Integration: CPU forward smoke for `task=sem_seg` (shapes only, no weights needed).

## 9. Phasing

1. **Core** (architecture + loss + dataset + train/validator): train a sem_seg model, see
   mIoU improve, visualize eval overlays. This is the reviewable milestone. Baseline = plain
   resize, out_stride 4; first ablation = `scale_jitter: [0.75, 1.5]`.
2. **Inference**: `Torch_model` sem_seg path + `infer.py` overlay/PNG output.
3. **Export + bench**: ONNX/OV/TRT/etc. graph, parity swap, bench mIoU.

## 10. Touch-point summary

| File | Change |
|---|---|
| `config.yaml` | `sem_seg` value + comment, `train.sem_seg` block |
| `src/d_fine/configs.py` | `SemSegCriterion.weight_dict` in base_cfg |
| `src/d_fine/dfine.py` | `build_model`/`build_loss` pick decoder+loss by task; generalize nano low-level gate |
| `src/d_fine/arch/dfine_decoder.py` | new `SemSegDecoder` (reuses `MaskDecoder` + neck + aux) |
| `src/d_fine/sem_seg_criterion.py` | new CE+Dice+aux loss |
| `src/dl/dataset.py` | `SemSegDataset` + `sem_seg_collate_fn` + Loader dispatch |
| `src/dl/validator.py` | `SemSegValidator` (pixel confusion → mIoU, original-res protocol) |
| `src/dl/train.py` | task branch, decision_metrics=[mIoU], sem_seg eval path |
| `src/infer/torch_model.py` | task-aware ctor, sem_seg postprocess → `out["sem_seg"]` |
| `src/dl/infer.py` | sem_seg overlay + label-map PNG output |
| `src/dl/export.py` | skip detection postproc, fused int32 argmax output + parity swap |
| `src/dl/bench.py` | sem_seg mIoU branch |
| `tests/` | sem_seg unit + smoke |
| `CLAUDE.md` | document the third task |

## Open questions

- Overlay palette: reuse `Visualizer` colors or a dedicated sem_seg palette? (cosmetic)
- Loss resolution: upsample logits to full res then CE (simpler, accurate, small C) — assumed.
  Switch to 1/4-res CE only if memory bites.
- `ignore_negatives` in `split.py` for sem_seg should check `masks/` not `labels/`.
- Thin structures (wires/poles) at drone GSD: if they matter and out_stride 2 isn't enough,
  the later knob is pulling a stride-4 backbone feature into the fuser (the nano low-level
  mechanism generalizes). Not in v1.
