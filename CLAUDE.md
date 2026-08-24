# CLAUDE.md — D-FINE-seg agent guide

Commands, paths, config keys here are exact. Deeper docs when relevant: README.md (user-facing), CHANGELOG.md,
scripts/regression_test.py (its module docstring is the whole guide to proving a training/export change is safe).

## 1. What this repo is
Detection + instance + semantic segmentation on D-FINE. One Hydra config (`config.yaml`) drives
split → train → export → bench → infer; `task: detect | segment | sem_seg`, `model_name: n|s|m|l|x`.
Released weights auto-download from HF (`ArgoSA/D-FINE-seg`): `dfine_<size>_{coco,obj2coco}.pt` (detect) /
`dfine_seg_<size>_coco.pt` (mask tasks, includes the trained MaskDecoder). `train.pretrained_dataset`
(coco|obj2coco) picks the detect file; mask tasks must point `train.pretrained_model_path` at `dfine_seg_*`
— `dfine init --task segment|sem_seg` does that, the live config does not (non-standard paths must exist
on disk). **No released sem_seg weights** — the API refuses `load_model(…, task="sem_seg")`; fine-tune from
`dfine_seg_*` (backbone+encoder+fuser transfer; neck/classifier train from scratch).

## 2. Layout
```
config.yaml       # live dev config; make targets = `uv run dfine <cmd>` (`dfine main` / `make` = train→export→bench)
dfine_seg/
  api/            # load_model()/read_image(), checkpoint auto-detect — torch-only
  app/            # cli.py (`dfine`), demo.py (Gradio, [demo] extra)
  config/         # default.yaml (`dfine init` template) + resolve.py (config discovery)
  etl/            # split, yolo2coco, coco2yolo, sam_labels, … (`python -m dfine_seg.etl.<name>`)
  dl/             # train, export, bench, infer, validator, ov_int8, trt_int8, …
  infer/          # standalone backend wrappers (torch/onnx/ov/trt/coreml/litert) users copy out
  model/          # arch, losses, matcher
  viz.py          # Visualizer (public) + sem_seg palette; drawing for API, CLI, demo, training
tests/            # pytest, zero training data needed
scripts/          # research/ops helpers (run_candidate.py, promote.py, regression_test.py, …)
```

## 3. Environment
- Python 3.11–3.13, CUDA 12.x; `uv sync` installs everything editable (`dev` → `dfine-seg[all]`: `[export]`
  onnx/ov/coreml, `[trt]` Linux-only, `[label]` transformers/SAM3, `[demo]` gradio, `[extra]`). **litert is
  paused** — litert-torch caps torch<2.13; re-add when it catches up.
- **Never add an unknown key to `[tool.uv]`.** uv discards the *entire* table on a parse error with only a
  warning, silently dropping the `dependency-metadata` tensorrt-cu12 pin + `environments` (the pair keeping
  one `uv.lock` portable across dev mac + lab box). That once flipped the lock to tensorrt-cu13.
- Ruff (100 cols, `ruff==0.15.20`) is CI-enforced — `uv run ruff check . && uv run ruff format .` before
  finishing edits. `make build` → `uv build` → `dist/`; nothing is published, the user publishes.

## 4. Configuration
Every command is Hydra — any key overrides on the CLI: `dfine train exp_name=my model_name=s train.batch_size=12`.
- Discovery ([dfine_seg/config/resolve.py](dfine_seg/config/resolve.py)): `$DFINE_SEG_CONFIG_DIR` → cwd →
  repo root. Deliberately **no fallback to the packaged template**; pip users run `dfine init` (`--task`,
  `--model`, `-d`, `--force`).
- Root `config.yaml` (live) and [dfine_seg/config/default.yaml](dfine_seg/config/default.yaml) (the `dfine init`
  template) are in lockstep — `tests/unit/test_config_template.py` fails on drift (`ALLOWED_VALUE_DIFFS` =
  keys allowed to differ). Presets in `configs/` are gitignored — never reference them in shipped docs.
- Fields to know: `train.root/data_path/path_to_save`, `train.coco_dataset`, `train.pretrained_dataset`/
  `pretrained_model_path`, `train.label_to_name`, `train.img_size`, `train.keep_ratio`, `train.in_channels`
  (**3 or 4 only**; 4 = RGB+extras, `.npy` stacks), `train.conf_thresh/iou_thresh`, `train.ddp.{enabled,n_gpus}`,
  `train.lrs.<size>`, `export.formats`, `bench.formats`.

## 5. Data
**YOLO (default):** `images/` + `labels/<stem>.txt` in `<data_path>`; detect `cls xc yc w h`, segment
`cls x1 y1 … xN yN` (normalized). **sem_seg:** `labels/*.png` uint8, pixel = class id; `sem_seg.ignore_index`
(255) excluded from loss/mIoU and used as pad fill; `label_to_name` covers every class incl. background;
`coco_dataset: True` rejected. **COCO:** `images/` + `train.json`/`val.json`(/`test.json`) with
`coco_dataset: True`; a single `coco.json` → `make split` splits it by image into those JSONs.
`make split` → `train/val(/test).csv` (YOLO) or `.json` (COCO). Inputs: 3-ch `.jpg/.png` (BGR) or 3/4-ch
`.npy` (RGB+extras); wrappers take `bgr: bool = True`. `task=segment` needs real polygons — bbox-only
COCO/JSON raises `_assert_has_polygons`.
**SAM3:** `python -m dfine_seg.etl.sam_labels /abs/images --prompt person --format coco --task segment`
writes `coco.json` (or YOLO txts) into `<src>_labels/`; repeat `--prompt` for multi-class ("car, person" in one flag also works). Detect labels SAM3's box head; segment measures
boxes off the written polygons (never raw mask extent). Needs `[label]` (in `uv sync`); gated `facebook/sam3`
is cache-only — `--model <local snapshot>` or `HF_HUB_OFFLINE=1`.

## 6. Training
```bash
make train    # == dfine train; DDP auto-launches via torchrun when train.ddp.enabled=True (batch per GPU)
dfine train exp_name=x model_name=s task=detect train.epochs=30   # overrides
```
- Default optimizer is **Muon** (`use_muon: True`: enc/dec attn+MLP matrices → Muon, rest → `aux_optimizer:
  adan`). `batch_size: -1` auto-picks from free VRAM (CUDA only). `freeze_except_mask: True` trains only the
  MaskDecoder (segment only). `max_walltime_min` caps a run, best epoch kept.
- **No resume flag** — fine-tune via `train.pretrained_model_path` (`strict=False`; stem auto-inflated for
  4-channel models; mask tasks must point at `dfine_seg_*` or MaskDecoder starts random).
- AMP defaults to **bf16** (`train.amp_dtype`), no NaN guard — if a run diverges apply gotcha 8. WandB on by
  default (`train.use_wandb`), project = `project_name`.

Outputs under `${train.path_to_save}` (= `${train.root}/output/models/<exp_name>_<date>`): `model.pt`
(**best** by `train.decision_metrics`; EMA weights when `use_ema`; use for infer/export/bench), `last.pt`
(final epoch, not a resume point — no optimizer/EMA state), frozen `config.yaml`, `train_log.txt`,
`extended_metrics.csv` (its `optimal_thresh` is informational — bench runs at `train.conf_thresh`).

## 7. Inference
- `make infer` / `dfine infer` — images + videos from `train.path_to_test_data`, checkpoint
  `${train.path_to_save}/model.pt`; outputs under `${train.infer_path}` (**wiped each run**): `images/`,
  `labels/` (YOLO txt), `crops/` (`infer.to_crop`), `<stem>_tracked.mp4` (videos, `infer.to_track`, ByteTrack;
  defaults in [dfine_seg/dl/infer.py](dfine_seg/dl/infer.py), override via top-level `track:`; fresh tracker
  per video). sem_seg: overlay + grayscale label PNGs; crops/txt/tracking skipped, videos become
  `<stem>_sem_seg.mp4`.
- Config-free tools: `dfine predict <size|path> <image|dir> [--task --conf --device -o out/]`; `dfine demo`
  binds **0.0.0.0** by default — the Model panel loads any path the browser sends, so `--host 127.0.0.1`
  for local-only.

Wrapper contract: `list[dict]` with `boxes/scores/labels` (+ `masks` `[N,H,W]` for segment); sem_seg returns
`out["sem_seg"]` — uint8 `[H,W]` label map at original resolution.

## 8. Export & bench
```bash
make export    # builds export.formats (null = onnx/tensorrt/openvino/coreml; litert only when named).
               # Knobs: export.half, max_batch_size, dynamic_input. export.from_pretrained exports released
               # weights without training; refuses on label_to_name class-count mismatch.
make bench     # benches bench.formats against val/test at train.conf_thresh; sem_seg → mIoU+pixel_acc.
```
Artifacts sit next to `model.pt`: `model.onnx` (postprocessor fused), `model.engine` (TRT, GPU-specific,
Linux), `model.xml/.bin` (OpenVINO, raw head), `model.mlpackage` (+int8), `parity.csv`. **Parity**
(`export.parity: True`): cosine over sorted top-K scores vs torch, warn-gated ≥ 0.99 (0.90 INT8); sem_seg
uses per-pixel argmax agreement. sem_seg ships one fused graph per backend (logits → bilinear ×4 → argmax →
int32 `[B,H,W]` at input resolution; wrappers NEAREST-resize to original). INT8: `make ov_int8` (NNCF
accuracy-aware, `export.ov_int8_max_drop`), `make trt_int8`. Also: `dfine test-batching`, `dfine check-errors`.

## 9. Public API / packaging invariants
- `from dfine_seg import load_model, read_image, Visualizer` (also `pretrained_path`, `SIZES`, `TASKS`).
  **`load_model` is a factory, not a wrapper** — returns the same `TorchModel`/`TRTModel`/… you'd build by
  hand (backend by file suffix, size string → HF weights, kwargs verbatim, `.names` attached). Don't
  reintroduce a wrapping class; never force outputs to `.cpu()`. `task=` forwards only for `.pt` — graph
  artifacts carry the task. `read_image`: BGR for `.jpg/.png`, RGB for `.npy`/PIL (pass `bgr=False`).
- **`Visualizer(model)` is the one drawing entry point** — `vis(img, out)` dispatches boxes/masks vs the
  sem_seg overlay (`classes_from_model` reads count+names off the wrapper, 80 when a graph carries neither;
  palette built lazily, once). Takes 3-channel BGR only: callers slice/flip `.npy` stacks themselves
  (`cli.py`, `dl/infer.py`). `Visualizer(n_classes=…, class_names=…)` still works for the training paths.
- **`import dfine_seg` must stay torch-only.** hydra/wandb/albumentations/matplotlib/pandas/sklearn/
  torchmetrics must not be reachable from API module scope (`tests/integration/test_light_import.py` +
  `core-install` CI job).
- **Checkpoints** = `{"model": state_dict, "meta": {…}}` (`save_checkpoint`; meta from `ckpt_meta(cfg)`:
  version/model_name/task/num_classes/in_channels/label_to_name/img_size/keep_ratio). Plain python only in
  meta (`weights_only=True`); read only via `unwrap_checkpoint` (envelope / bare state_dict / legacy
  `{"ema":{"module":…}}` — fix a bypassing load site, don't add a second reader); no optimizer/EMA state.
- **Architecture always comes from the weights**, never meta ([api/ckpt.py](dfine_seg/api/ckpt.py)):
  num_classes/task/in_channels from key shapes, model_name from a fingerprint table pinned by a slow test;
  meta breaks ties only for unknown future sizes. Preprocessing: explicit arg → ckpt meta → sidecar
  `config.yaml` → 640/False; the Hydra commands pass `img_size` (+`task`/`keep_ratio`/`in_channels`)
  explicitly, so **the live config wins** there — keep it in sync with training. Graph exports carry no
  metadata/class names.
- `n_outputs` is gone from onnx/trt/coreml (fused graphs emit labels); the position belongs to `conf_thresh`
  — onnx/coreml reject a scalar outside [0,1] naming the removal. LiteRT keeps optional `n_outputs` (label
  decode); `OVModel` reads it off the graph.

## 10. Testing
```bash
make test-fast   # unit + CPU smoke, seconds
make test        # + slow regression (dfine_s_coco.pt in pretrained/, fixtures + baseline.json in assets/)
```
Markers: `slow`, `gpu` (auto-skip without CUDA). `tests/unit/` pins pure helpers, no weights. After a
deliberate model change: `uv run python -m tests.generate_fixtures` (writes labels + `baseline.json` into
`tests/assets/`; commit them). New fixture images: drop in `tests/assets/`, re-run the bootstrap.

## 11. Gotchas
1. **Hydra interpolation:** `${train.lrs.${model_name}.base_lr}` follows a `model_name` override
   automatically — don't also override LRs unless intentional.
2. **`exp` is timestamped:** training nests under `<exp_name>_<date>`; export/bench/infer resolve it to the
   newest matching run dir automatically (`get_latest_experiment_name`).
3. **COCO vs YOLO is exclusive** — flipping `train.coco_dataset` without matching files fails in the loader
   (COCO also rejected for sem_seg).
4. **`label_to_name` must be 0-indexed and contiguous.**
5. **`mosaic_augs.mosaic_prob: null` = task default** (0.8 detect, 0.5 segment/sem_seg); a number wins. For
   instance segment, lower it toward 0 if masks look wrong.
6. **Decision metrics auto-swap:** `mAP_50` → `mAP_50_mask` for segment; sem_seg forces `mIoU`.
7. **DDP rank-0 writes everything** — logs, checkpoints, wandb gated to rank 0.
8. **NaN recipe** (bf16 run diverging; from [notes.md](notes.md)): lower both LRs; `weight_decay:
   0.000125`–`0.00025`; `betas: [0.9, 0.98]`; `label_smoothing: 0.1`; `mosaic_scale: [0.5, 1.4]` if
   object-sparse.
9. **Multichannel = `.npy`, never TIFF** (cv2 mangles 4-ch TIFFs; `.npy` byte-faithful, ~25× faster).
   `train.in_channels` is **3 or 4 only**; at 4, stem freeze auto-bypasses so inflated extra-channel
   weights train (`freeze_at` in [dfine_seg/model/configs.py](dfine_seg/model/configs.py)).
10. **Run TensorRT engines at batch 1** — TRT 10.13.3.9 batched engines are slot-dependent; identical
    images in one batch return different results
    ([NVIDIA/TensorRT#4813](https://github.com/NVIDIA/TensorRT/issues/4813)); batch 1 is exact and benched
    faster anyway.
11. **Don't redo measured-and-rejected optimizations.** (a) segment TRT: engine-side postprocess fusions
    (mask_feat emit, fp16 masks output, NMS-in-graph) were all built, timed, rejected — wins are client-side,
    already live in every `/infer` wrapper (fp16 interpolate gated on `is_cuda`, no mask `clamp_`,
    `.view(uint8)`, separable box crop); the training-eval copies in `dl/utils.py` are deliberately not
    ported. (b) sem_seg: bilinear-upsample-before-argmax rejected (+0.004 mIoU, ~+20% TRT latency);
    argmax → NEAREST stays.

## 12. Code style & version control
- **Be concise** — as little code as possible; comments short, core info only, match the file's density.
- **Never `git commit`, push, branch, or open PRs without the user explicitly asking in that request.** Leave
  changes uncommitted for the user to review; report what changed and where. This overrides any default
  "ship it" / background-job workflow.
