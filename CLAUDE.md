# CLAUDE.md — D-FINE-seg agent guide

Reference for AI agents working in this repo. Keep it open and follow it literally — commands, paths, and config keys are exact.

## 1. What this repo is

D-FINE-seg is a detection + instance segmentation framework built on D-FINE. A single config (`config.yaml`, Hydra-based) drives the whole pipeline: dataset split → train → export → bench → infer. One task flag (`task: detect` or `task: segment`) switches between object detection and instance segmentation.

Main supported model sizes: `n`, `s`, `m`, `l`, `x`. Pretrained weights live in `pretrained/` (`dfine_<size>_coco.pt` and `dfine_<size>_obj2coco.pt`).

## 2. Layout

```
config.yaml                  # main Hydra config (edit this for most tasks)
Makefile                     # thin wrappers around python -m src.dl.*
pretrained/                  # dfine_{n,s,m,l,x}_{coco,obj2coco}.pt — must exist before training
src/
  etl/                       # dataset prep: split, yolo2coco, coco2yolo, polys2bbox, …
  dl/                        # train.py, export.py, bench.py, infer.py, validator.py, ov_int8.py, …
  d_fine/                    # model architecture (backbone, encoder, decoder, matcher, losses)
  infer/                     # multi-backend inference wrappers (torch, onnx, ov, trt, coreml, litert)
```

## 3. Environment

- Python 3.11–3.13, PyTorch 2.9, CUDA 12.x. Dependencies live in [pyproject.toml](pyproject.toml); [uv.lock](uv.lock) is the source of truth for versions.
- Install with `uv sync` (creates `.venv/`). All Makefile targets shell out via `uv run`, so no manual activation is needed for `make train` / `make bench` / etc. For ad-hoc commands either prefix with `uv run` or activate the venv (`source .venv/bin/activate`).
- Platform-specific deps are gated by markers in `pyproject.toml`: `tensorrt` installs on Linux only. `coremltools` ships wheels for both platforms (Linux can run the converter for `make export`, even though the CoreML runtime itself is macOS-only). `uv.lock` covers both so the same lockfile works on the dev mac and the lab box.
- Pretrained weights auto-download from Hugging Face (`ArgoSA/D-FINE-seg`) into `pretrained/` on first use via `ensure_pretrained` in [src/d_fine/utils.py](src/d_fine/utils.py). Triggered from `build_model` in [src/d_fine/dfine.py](src/d_fine/dfine.py) only when the filename matches `dfine_<size>_<dataset>.pt`; custom checkpoint paths still raise `FileNotFoundError` if missing.

## 4. Configuration model

All CLI commands use Hydra, so any config key is overridable on the command line with dotted paths:

```bash
python -m src.dl.train exp_name=my_exp model_name=s train.batch_size=12 train.epochs=50
```

Key top-level fields in [config.yaml](config.yaml):

| Field | Meaning |
|---|---|
| `project_name` | WandB project name |
| `exp_name` | Experiment name (outputs nest under `<exp_name>_<date>`) |
| `model_name` | `n` / `s` / `m` / `l` / `x` |
| `task` | `detect` or `segment` |
| `train.root` | Absolute project root (dataset + outputs live here) |
| `train.data_path` | Dataset dir — `${train.root}/data/dataset` by default |
| `train.coco_dataset` | `False` → YOLO-style; `True` → COCO JSON |
| `train.pretrained_dataset` | `coco` or `obj2coco` |
| `train.pretrained_model_path` | Path to init weights (swap this to fine-tune from a custom checkpoint) |
| `train.path_to_save` | Where `model.pt`, `last.pt`, logs, and configs land |
| `train.path_to_test_data` | Folder of images/videos for `infer.py` |
| `train.label_to_name` | 0-indexed, contiguous class map |
| `train.ddp.enabled` / `train.ddp.n_gpus` | Multi-GPU switch |
| `train.conf_thresh` / `train.iou_thresh` | Detection thresholds |

LRs are indexed by model size under `train.lrs.<size>.{backbone_lr, base_lr}`.

Preset dataset configs in [configs/](configs/) can be used as templates — copy one to `config.yaml` and edit paths / classes.

## 5. Dataset preparation

### 5.1 YOLO layout (default)

```
<train.data_path>/
  images/   # .jpg/.png/.jpeg
  labels/   # .txt — same stem as image
```

Label format:
- **detect**: `class_id xc yc w h` (normalized, cxcywh)
- **segment**: `class_id x1 y1 x2 y2 … xN yN` (normalized polygon)

Supported input types: 3-channel `.jpg`/`.png` (BGR, `cv2.imread`), 3-channel `.npy` (RGB, `np.load`), 4-channel `.npy` (RGB+extras, e.g. RGB+thermal). Inference wrappers' `__call__` takes a `bgr: bool = True` flag — repo callers pass `bgr=False` for `.npy` reads.

Generate splits:

```bash
make split        # == python -m src.etl.split
```

Produces `train.csv`, `val.csv` (and `test.csv` if `split.val_split < 1 - split.train_split`) inside `train.data_path`. Ratios live under the top-level `split:` section in `config.yaml`.

### 5.2 COCO layout

```
<train.data_path>/
  images/
  train.json
  val.json
  test.json         # optional
```

Then set `train.coco_dataset: True`. No `make split` needed.

### 5.3 Conversion / cleanup utilities

In [src/etl/](src/etl/): `yolo2coco.py`, `coco2yolo.py`, `polys2bbox.py`, `png_mask_to_yolo.py`, `remove_dups.py`, `clean_csv.py`, `split_from_yolo.py`. Run as `python -m src.etl.<name>`.

## 6. Training

### 6.1 Single-GPU

```bash
make train
# or explicit:
python -m src.dl.train
# with overrides:
python -m src.dl.train exp_name=fine_s model_name=s task=detect train.batch_size=12 train.epochs=30
```

### 6.2 Multi-GPU (DDP)

Set `train.ddp.enabled: True` and `train.ddp.n_gpus: N` in `config.yaml`, then:

```bash
make train
# Makefile auto-detects DDP and launches: torchrun --nproc_per_node=N --master_port=29500 -m src.dl.train
```

`batch_size` is **per GPU** in DDP. Effective batch = `batch_size × n_gpus × b_accum_steps`.

### 6.4 Outputs

Under `${train.path_to_save}` (= `${train.root}/output/models/<exp>`):
- `model.pt` — **best** checkpoint by `train.decision_metrics` (use this for inference/export)
- `last.pt` — last-epoch checkpoint, used only for NaN recovery
- `config.yaml` — frozen snapshot of the run's config
- `train_log.txt` — loguru log
- Confusion matrices, per-class metric CSVs, F1-vs-threshold plots, and eval visualizations
- `extended_metrics.csv` — per-class metrics plus an `optimal_thresh` column: the conf threshold
  that maximizes f1, found by a sweep over the *unfiltered* predictions during the final eval (val
  and test rows each get their own). `make bench` reads the **val** row and benches at that
  threshold (the prod operating point), instead of a fixed 0.5.

### 6.5 Logging

- WandB on by default (`train.use_wandb: True`); disable by setting to `False` if agent is offline
- `WANDB_PROJECT` env var overrides `project_name` if set

## 7. Fine-tuning / resume

There is **no built-in resume flag**. To continue training from a checkpoint:

```bash
python -m src.dl.train \
  exp_name=continue_run \
  train.pretrained_model_path=/abs/path/to/previous/model.pt
```

i.e. point `pretrained_model_path` at any `.pt` with matching architecture. Weights load non-strictly (`strict=False`), so head mismatches (e.g., fine-tuning a COCO-pretrained model on your own classes) are tolerated.

**Automatic NaN recovery** is built in: if 10 consecutive batches produce non-finite loss, the run reloads `last.pt` and continues. If NaNs persist, apply the recipe in section 11.

## 8. Inference

One entrypoint handles both images and videos based on file extension in `train.path_to_test_data`.

```bash
make infer
# or:
python -m src.dl.infer
# with overrides:
python -m src.dl.infer train.path_to_test_data=/abs/path/to/folder infer.to_crop=False
```

Supported inputs: `.jpg`, `.png`, `.jpeg`, `.mp4`, `.avi`, `.mov`, `.mkv`.

Outputs land under `${train.infer_path}`:
- `images/` — annotated frames (boxes + masks + labels)
- `labels/` — YOLO-format predictions per frame
- `crops/` — per-object crops (when `infer.to_crop: True`, padded by `infer.paddings.{w,h}`)
- `<stem>_tracked.mp4` — for videos, when `infer.to_track: True` (default): persistent IDs via ByteTrack ([src/infer/byte_track.py](src/infer/byte_track.py)). Defaults are baked into [src/dl/infer.py](src/dl/infer.py); override any of them via a top-level `track:` block (e.g. `track.track_buffer=60`). A fresh tracker is instantiated per video so IDs don't bleed across clips.
- `labels.txt` — classes seen across the run

Checkpoint used: `${train.path_to_save}/model.pt`. Threshold knobs: `train.conf_thresh`, `train.iou_thresh`. NMS IoU is set inside [src/infer/torch_model.py](src/infer/torch_model.py).

For interactive threshold tweaking, the Gradio UI in [demo/](demo/) exposes a threshold slider.

Important to note: inference wrappers under /infer are standalone scripts that are usually taken with the model file and used in users' applications, outside of this repo.

## 9. Benchmarking

```bash
make bench        # == python -m src.dl.bench
```

Runs the val/test set through each backend listed in `formats_to_bench` inside [src/dl/bench.py](src/dl/bench.py) and reports per-backend latency (ms/image, CUDA-synced, warmup skipped) and F1 / mAP vs GT. Bench runs at the **val-optimal conf threshold** read from `extended_metrics.csv` (not a fixed 0.5); it fails loudly if that file/column is missing (run training/eval first). Edit `formats_to_bench` to include/exclude `"torch"`, `"onnx"`, `"openvino"`, `"tensorrt"`, `"coreml"`, `"litert"`. The exported artifact for each backend must already exist (run `make export` first).

Related:
- `python -m src.dl.test_batching` — sweeps batch sizes, writes `batched_infer.csv`
- `python -m src.dl.check_errors` — dumps FP/FN mismatches against GT

## 10. Export / conversion

```bash
make export       # == python -m src.dl.export
```

Produces, under `${train.path_to_save}`:
- `model.onnx` (always)
- `model.engine` — TensorRT (skipped on macOS; engine is GPU-specific, rebuild on target hardware)
- `model.xml` + `model.bin` — OpenVINO
- `model.mlpackage` + `model_int8.mlpackage` — CoreML (converter runs on Linux + macOS; runtime/`make bench` on the CoreML backend is macOS-only)
- `model.tflite` + `model_int8.tflite` — LiteRT

Knobs under `export:` in `config.yaml`: `half` (FP16), `max_batch_size`, `dynamic_input`.

ONNX has the D-FINE postprocessor fused into the graph; OpenVINO exports the raw head (postprocess separately).

### 10.1 INT8 quantization

```bash
make ov_int8      # OpenVINO INT8 via NNCF, accuracy-aware (can take hours)
make trt_int8     # TensorRT INT8 calibration
```

OpenVINO path respects `ov_int8_max_drop` (default 0.02 F1 drop allowed).

## 11. Testing

Pytest suite under [tests/](tests/), zero training data required.

```bash
make test-fast   # ~5s on Mac, all unit tests + CPU forward smoke
make test        # full suite incl. the ~5s CPU pretrained accuracy regression
```

Layout:
- `tests/unit/` — pin pure helpers (box conversions, IoU, RLE, letterbox, NMS, matcher, losses, Validator, ETL). No model or weights loaded.
- `tests/integration/test_cpu_forward.py` — shapes + a loose CPU forward latency ceiling. GPU variant is marked `@pytest.mark.gpu` and auto-skips when CUDA isn't present.
- `tests/integration/test_pretrained_accuracy.py` (marked `slow`) — loads `dfine_s_coco.pt` on CPU, runs through `Torch_model` on the source images in `tests/assets/`, asserts `mAP_50 ≥ baseline_min` from `tests/assets/baseline.json`. Catches any silent drift in the model arch / weights loader / postprocess / letterbox / NMS / Validator math.

Pytest markers (declared in [pyproject.toml](pyproject.toml)): `slow`, `gpu`. Use `-m "gpu"` on the lab box to target GPU tests, `-m "not slow"` to skip the pretrained regression on a flaky network.

Fixture layout — everything lives flat in `tests/assets/`:

```
tests/assets/
  park_gen.jpg              # downscaled to ≤1024px long side, committed
  park_gen.txt              # YOLO labels (regenerated by bootstrap)
  west_cost_gen.jpg
  west_cost_gen.txt
  baseline.json             # pinned mAP_50_min / mAP_50_95_min
```

To add a new image: drop a `.png` / `.jpg` into `tests/assets/`, re-run the bootstrap (normalizes to `.jpg`, downscales to ≤1024px, removes the source if it wasn't already `.jpg`), commit the new label + updated `baseline.json`. The test picks up any image that has a matching `<stem>.txt` next to it.

Regenerate the accuracy regression baseline after a deliberate model change:

```bash
uv run python -m tests.generate_fixtures
# Runs dfine_s_coco.pt on every image in tests/assets/, writes the
# high-confidence predictions as YOLO labels next to each image, and pins a
# fresh baseline.json. Commit the result.
```

## 12. Gotchas — read before big changes

1. **Hydra interpolation order.** Keys like `${train.lrs.${model_name}.base_lr}` resolve `${model_name}` first. If you override `model_name`, the nested LR lookup follows automatically — don't also override LRs unless intentional.
2. **`exp` is timestamped.** `exp: ${exp_name}_${now_dir}` → outputs always nest under a dated folder.
3. **Pretrained weights auto-download from HF on first use** for the standard `dfine_<size>_<dataset>.pt` filenames. Custom `train.pretrained_model_path` values (e.g. fine-tuning checkpoints) are not fetched — those must exist on disk.
4. **COCO vs YOLO is mutually exclusive.** Flipping `train.coco_dataset` without having the matching files on disk will fail in the loader.
5. **`label_to_name` must be 0-indexed and contiguous.**
6. **`task: segment` disables mosaic-friendliness.** Mosaic augmentation is not recommended for segmentation — lower `mosaic_augs.mosaic_prob` toward 0 if masks look wrong.
7. **Decision metrics swap for segment.** `mAP_50` becomes `mAP_50_mask` automatically when `task: segment`.
8. **NaN recipe** (from [notes.txt](notes.txt), applied when auto-recovery keeps firing):
   - Lower both `backbone_lr` and `base_lr`
   - `train.weight_decay: 0.000125` (or even `0.00025`)
   - `train.betas: [0.9, 0.98]`
   - `train.label_smoothing: 0.1`
   - `train.mosaic_augs.mosaic_scale: [0.5, 1.4]` if dataset is object-sparse
9. **DDP rank-0 writes everything.** Don't assume per-rank directories; logs, checkpoints, and WandB calls are gated to rank 0.
10. **`model.pt` is best, `last.pt` is for recovery only.** Always use `model.pt` for inference, export, and bench.
11. **Multi-channel images live in `.npy`, not TIFF.** `cv2.imread(IMREAD_UNCHANGED)` is not byte-faithful for 4-channel TIFFs — it treats channel 4 as alpha, swaps the first three per the photometric tag, and pre-multiplies values, so any TIFF from a non-cv2 writer is silently mangled. `.npy` is byte-faithful and ~25× faster to read.
12. **Stem freeze auto-bypassed for inflated stems.** `freeze_at >= 0` in [src/d_fine/configs.py](src/d_fine/configs.py) only freezes the stem when `train.in_channels == 3`; for `in_channels > 3` the freeze is skipped so the inflated extra-channel weights can train.

## 13. Quick reference

| Task | Command |
|---|---|
| Prepare YOLO splits | `make split` |
| Train (single GPU) | `python -m src.dl.train exp_name=<name> model_name=<size>` |
| Train (multi-GPU) | set `train.ddp.enabled=True`, `train.ddp.n_gpus=N`, then `make train` |
| Fine-tune from checkpoint | `python -m src.dl.train train.pretrained_model_path=/abs/path/model.pt exp_name=<new>` |
| Infer on folder (images or video) | `python -m src.dl.infer train.path_to_test_data=/abs/path` |
| Export all formats | `make export` |
| Benchmark exports | `make bench` |
| Full pipeline | `make` (train → export → bench) |
| Find best batch size | `python -m src.dl.test_batching` |
| Inspect FP/FN | `python -m src.dl.check_errors` |
| OpenVINO INT8 | `make ov_int8` |
| TensorRT INT8 | `make trt_int8` |
| Run unit + smoke tests | `make test-fast` |
| Run full test suite | `make test` |
| Regenerate accuracy baseline | `uv run python -m tests.generate_fixtures` |

## 14. Code style

- **Be consice** - write as little code as possible to achieve the goal.

- **Keep comments short — core info only.** Prefer a single terse line. Don't restate what the code already says or narrate rationale at length; capture just the non-obvious fact. Match the existing comment density of the surrounding file.
