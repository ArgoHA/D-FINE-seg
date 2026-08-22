# CLAUDE.md — D-FINE-seg agent guide

Commands, paths, and config keys here are exact — follow them literally. Deeper docs, read when relevant:
README.md (user-facing), CHANGELOG.md, scripts/regression_test.py (its module docstring is the whole guide to proving a training or export path change is safe).

## 1. What this repo is

Detection + instance segmentation + semantic segmentation framework built on D-FINE. One Hydra
config (`config.yaml`) drives the whole pipeline: split → train → export → bench → infer.
`task: detect | segment | sem_seg` switches the task, `model_name: n|s|m|l|x` the size.
Pretrained weights auto-download from HF (`ArgoSA/D-FINE-seg`) into `pretrained/` for the exact
filenames `dfine_<size>_{coco,obj2coco}.pt` / `dfine_seg_<size>_coco.pt` (seg = includes trained
MaskDecoder); custom `train.pretrained_model_path` values must exist on disk.

## 2. Layout

```
config.yaml            # live dev config; Makefile targets are thin wrappers over `uv run dfine <cmd>`
dfine_seg/
  api/                 # load_model()/read_image(), checkpoint auto-detect — torch-only
  app/                 # cli.py (`dfine` console script), demo.py (Gradio, [demo] extra)
  config/              # default.yaml (`dfine init` template) + resolve.py (config discovery)
  etl/                 # split, yolo2coco, coco2yolo, sam_labels, … (`python -m dfine_seg.etl.<name>`)
  dl/                  # train, export, bench, infer, validator, ov_int8, trt_int8, …
  infer/               # standalone backend wrappers (torch/onnx/ov/trt/coreml/litert) — users copy
                       # these out of the repo with the model file, keep them self-contained
  model/               # arch, losses, matcher
  viz.py               # Visualizer + sem_seg palette, shared by dl/ and app/
tests/                 # pytest, zero training data needed
scripts/               # research/ops helpers (run_candidate.py, promote.py, backfill_ckpt_meta.py, …)
```

## 3. Environment

- Python 3.11–3.13, CUDA 12.x. `uv sync` installs everything editable (the `dev` group depends on
  `dfine-seg[all]`). Ad-hoc commands: prefix with `uv run`.
- Core install (`pip install dfine-seg`) = torch inference **and** training. Extras: `[export]`
  (onnx/onnxruntime/openvino/nncf/coremltools), `[trt]` (Linux-only marker), `[label]`
  (transformers/SAM3), `[demo]` (gradio), `[extra]`, `[all]`. **litert is paused** — litert-torch
  caps torch<2.13; re-add when it catches up.
- **Never add an unknown key to `[tool.uv]`.** uv discards the *entire* table on a parse error
  with only a warning, silently dropping the `dependency-metadata` tensorrt-cu12 pin and
  `environments`. That once flipped the lock to tensorrt-cu13.
- `uv.lock` covers darwin + linux so the same lockfile serves the dev mac and the lab box.
- Ruff (line-length 100, `ruff==0.15.20`) is CI-enforced — run `uv run ruff check . && uv run ruff
  format .` before finishing edits.
- `make build` → `uv build` → `dist/`. Nothing is published; the user publishes and commits.

## 4. Configuration

Every command is Hydra, so any key overrides on the CLI:
`dfine train exp_name=my model_name=s train.batch_size=12 train.epochs=50`.

- Config discovery ([dfine_seg/config/resolve.py](dfine_seg/config/resolve.py)):
  `$DFINE_SEG_CONFIG_DIR` → cwd → repo root. Deliberately **no fallback to the packaged
  template**; pip users run `dfine init` (flags: `--task`, `--model`, `-d`, `--force`).
- Two configs in lockstep: root `config.yaml` (live) and
  [dfine_seg/config/default.yaml](dfine_seg/config/default.yaml) (what `dfine init` emits).
  `tests/unit/test_config_template.py` fails on drift; `ALLOWED_VALUE_DIFFS` lists keys allowed
  to differ. Local presets in `configs/` are **gitignored** — never reference them in shipped docs.
- Fields to know: `train.root/data_path/path_to_save`, `train.coco_dataset`,
  `train.label_to_name`, `train.img_size`, `train.keep_ratio`, `train.conf_thresh/iou_thresh`,
  `train.ddp.{enabled,n_gpus}`, per-size LRs under `train.lrs.<size>`, `export.formats`,
  `bench.formats`.

## 5. Data

**YOLO (default):** `<data_path>/images/` + `labels/` (same stem `.txt`). detect:
`cls xc yc w h` normalized; segment: `cls x1 y1 … xN yN` normalized polygon.
**sem_seg:** `labels/*.png`, single-channel uint8, pixel value = class id;
`train.sem_seg.ignore_index` (255) pixels excluded from loss/mIoU and used as pad fill;
`label_to_name` covers every pixel class incl. background; `coco_dataset: True` is rejected.
**COCO:** `images/` + `train.json`/`val.json`(/`test.json`) with `train.coco_dataset: True`; a
single `coco.json` is also accepted — `make split` splits it by image into standalone JSONs.

`make split` produces `train/val(/test).csv` (YOLO) or `.json` (COCO); ratios under `split:`.
Inputs: 3-ch `.jpg/.png` (BGR) or 3/4-ch `.npy` (RGB+extras); wrappers take `bgr: bool = True`.

**SAM3 pre-annotation:** `python -m dfine_seg.etl.sam_labels /abs/images --prompt person
--format coco --task segment` — the module docstring covers formats and the deliberate
box-source split (detect = SAM3 box head, segment = boxes measured off written polygons; never
derive boxes from raw mask extent). Needs the `[label]` extra (in `uv sync`); `facebook/sam3` is
gated but loads `local_files_only=True` from a cached snapshot.

## 6. Training

```bash
make train                    # == dfine train; DDP auto-launches under torchrun when
                              # train.ddp.enabled=True (batch_size is then per GPU)
dfine train exp_name=x model_name=s task=detect train.epochs=30    # overrides
```

- Optimizer default is **Muon** (`use_muon: True`: enc/dec attn+MLP matrices → Muon, rest →
  `aux_optimizer: adan`). `batch_size: -1` auto-picks from free VRAM. `freeze_except_mask: True`
  trains only the MaskDecoder (segment only). `max_walltime_min` caps a run, best epoch kept.
- **No resume flag.** Fine-tune by pointing `train.pretrained_model_path` at any matching `.pt`
  (loads `strict=False`, so head mismatches are fine).
- AMP defaults to **bf16** (`train.amp_dtype`); there is no NaN guard — if a run still diverges,
  apply the recipe in gotcha 8.
- WandB on by default (`train.use_wandb`); project name = `project_name`.

Outputs under `${train.path_to_save}` (= `${train.root}/output/models/<exp_name>_<date>`):
`model.pt` (**best** by `train.decision_metrics` — use for infer/export/bench), `last.pt`
(final epoch, not a resume point — no optimizer/EMA state), frozen `config.yaml`,
`train_log.txt`, plots, `extended_metrics.csv` (its `optimal_thresh` column is informational
only — bench runs at `train.conf_thresh`).

## 7. Inference

- `make infer` / `dfine infer` — images + videos from `train.path_to_test_data`, checkpoint
  `${train.path_to_save}/model.pt`. Outputs under `${train.infer_path}`: `images/` (annotated),
  `labels/` (YOLO txt), `crops/` (`infer.to_crop`), `<stem>_tracked.mp4` (videos,
  `infer.to_track`, ByteTrack; defaults in [dfine_seg/dl/infer.py](dfine_seg/dl/infer.py),
  override via top-level `track:` block; fresh tracker per video). sem_seg: palette overlays +
  grayscale label-map PNGs; crops/txt/tracking skipped, videos get `<stem>_sem_seg.mp4`.
- `dfine predict <size|path> <image|dir> [--task --conf --device -o out/]` — config-free
  inference for pip users.
- `dfine demo` — Gradio UI, no config needed; binds **0.0.0.0** by default (warned on launch:
  the Model panel loads any path the browser sends); `--host 127.0.0.1` for local-only.

Wrapper output contract: detection `list[dict]` with `boxes/scores/labels` (+ `masks` `[N,H,W]`
for segment); sem_seg returns `out["sem_seg"]` — uint8 `[H,W]` label map at original resolution.

## 8. Export & bench

```bash
make export    # builds export.formats (null = onnx/tensorrt/openvino/coreml; litert only when
               # named explicitly). Knobs: export.half, max_batch_size, dynamic_input.
make bench     # benches backends in bench.formats against val/test at train.conf_thresh;
               # sem_seg reports mIoU + pixel_acc. Artifacts must exist (export first).
```

Artifacts land next to `model.pt`: `model.onnx` (postprocessor fused), `model.engine` (TRT,
GPU-specific, Linux), `model.xml/.bin` (OpenVINO, raw head), `model.mlpackage` (+int8),
`parity.csv`. **Parity self-check** (`export.parity: True`): per-backend cosine over sorted
top-K scores vs torch, warn-gated at ≥ 0.99 (0.90 INT8); sem_seg uses per-pixel argmax
agreement. sem_seg exports one fused graph for every backend: logits → bilinear ×4 → argmax →
int32 `[B,H,W]` at input resolution; wrappers NEAREST-resize to original size.
INT8: `make ov_int8` (NNCF accuracy-aware, respects `export.ov_int8_max_drop`), `make trt_int8`.
Related: `dfine test-batching`, `dfine check-errors`.

## 9. Public API / packaging invariants

- `from dfine_seg import load_model, read_image` (also `pretrained_path`, `SIZES`, `TASKS`).
  **`load_model` is a factory, not a wrapper**: it returns the same `TorchModel`/`TRTModel`/…
  you'd construct by hand (backend picked by file suffix, size string → HF weights, kwargs
  passed through verbatim, `.names` attached). Do not reintroduce a class that wraps the
  wrappers; never force outputs to `.cpu()`. `task=` is forwarded only for `.pt` — graph
  artifacts carry the task in the graph.
- **`import dfine_seg` must stay torch-only.** hydra/wandb/albumentations/matplotlib/pandas/
  sklearn/torchmetrics are training deps and must not be reachable from API module scope.
  Enforced by `tests/integration/test_light_import.py` + the `core-install` CI job.
- **Checkpoints** are `{"model": state_dict, "meta": {…}}` (`save_checkpoint`; meta from
  `ckpt_meta(cfg)` — version/model_name/task/num_classes/in_channels/label_to_name/img_size/
  keep_ratio). Rules: plain python only in meta (`weights_only=True` loads); read only through
  `unwrap_checkpoint` (absorbs envelope / bare state_dict / legacy `{"ema":{"module":…}}`; fix a
  bypassing load site, don't add a second reader); no optimizer/EMA training state ever.
- **Architecture always comes from the weights**, never from meta ([api/ckpt.py](dfine_seg/api/ckpt.py):
  num_classes/task/in_channels from key shapes, model_name from a fingerprint table pinned by a
  slow test; meta breaks ties only for unknown future sizes). Preprocessing resolves explicit
  arg → ckpt meta → sidecar `config.yaml` → 640/False; the Hydra commands still pass
  `cfg.train.img_size` explicitly, so the live config wins there. Graph exports carry no
  metadata/class names.
- `n_outputs` is gone from onnx/trt/coreml wrappers (fused graphs emit labels); position now
  belongs to `conf_thresh` and a scalar outside [0,1] is rejected naming the removal. LiteRT
  keeps optional `n_outputs` (load-bearing for label decode); `OVModel` reads it off the graph.

## 10. Testing

```bash
make test-fast   # unit + CPU smoke, seconds
make test        # + slow pretrained-accuracy regression (dfine_s_coco.pt on tests/assets/)
```

Markers: `slow`, `gpu` (auto-skips without CUDA). `tests/unit/` pins pure helpers, no weights.
After a deliberate model change, regenerate the accuracy baseline:
`uv run python -m tests.generate_fixtures` (writes labels + `baseline.json`; commit them). New
fixture images: drop into `tests/assets/`, re-run the bootstrap, commit label + baseline.

## 11. Gotchas

1. **Hydra interpolation:** `${train.lrs.${model_name}.base_lr}` follows a `model_name` override
   automatically — don't also override LRs unless intentional.
2. **`exp` is timestamped:** outputs nest under `<exp_name>_<date>`.
3. **COCO vs YOLO is exclusive** — flipping `train.coco_dataset` without matching files fails in
   the loader.
4. **`label_to_name` must be 0-indexed and contiguous.**
5. **`mosaic_augs.mosaic_prob: null` = task default** (0.8 detect, 0.5 segment/sem_seg); a number
   wins. For instance segment, lower it toward 0 if masks look wrong.
6. **Decision metrics auto-swap:** `mAP_50` → `mAP_50_mask` for segment; sem_seg forces `mIoU`.
7. **DDP rank-0 writes everything** — logs, checkpoints, wandb are gated to rank 0.
8. **NaN recipe** (bf16 run diverging; from [notes.md](notes.md)): lower both LRs;
   `weight_decay: 0.000125`–`0.00025`; `betas: [0.9, 0.98]`; `label_smoothing: 0.1`;
   `mosaic_scale: [0.5, 1.4]` if object-sparse.
9. **Multi-channel images are `.npy`, never TIFF** — cv2 silently mangles 4-ch TIFFs (alpha
   premultiply + channel swap); `.npy` is byte-faithful and ~25× faster.
10. **Stem freeze auto-bypassed when `in_channels > 3`** so inflated extra-channel weights train
    (`freeze_at` in [dfine_seg/model/configs.py](dfine_seg/model/configs.py)).
11. **Run TensorRT engines at batch 1.** On TRT 10.13.3.9 batched engines are slot-dependent —
    identical images in one batch return different results
    ([NVIDIA/TensorRT#4813](https://github.com/NVIDIA/TensorRT/issues/4813)); batch 1 is exact
    and benched faster anyway.
12. **Don't redo measured-and-rejected optimizations.** (a) segment TRT: every engine-side
    postprocess fusion (mask_feat emit, fp16 masks output, NMS-in-graph) was built, timed, and
    rejected — the wins were client-side and already live in every `/infer` wrapper (fp16
    interpolate gated on `is_cuda`, no `clamp_`, `.view(uint8)`, separable box crop); the
    training-eval copies in `dl/utils.py` are deliberately not ported. (b) sem_seg: bilinear-
    upsample-before-argmax rejected (+0.004 mIoU, ~+20% TRT latency); argmax → NEAREST stays.

## 12. Code style & version control

- **Be concise** — write as little code as possible to achieve the goal.
- **Comments short, core info only** — capture the non-obvious fact, match the file's density.
- **Never `git commit`, push, branch, or open PRs without the user explicitly asking in that
  request.** Leave changes uncommitted for the user to review; report what changed and where.
  This overrides any default "ship it" / background-job workflow.
