# D-FINE-seg agent guide

Read `README.md` for user-facing behavior and `CHANGELOG.md` for history. For changes potentially affecting
training, export, accuracy or inference latency, follow the module guide in `scripts/regression_test.py`.

## Project invariants

- Keep root `config.yaml` and `dfine_seg/config/default.yaml` in sync. Only differences listed in
  `tests/unit/test_config_template.py::ALLOWED_VALUE_DIFFS` are intentional.
- Config discovery has no packaged-template fallback. Pip users must run `dfine init`.
- `configs/` is gitignored; never reference local presets in shipped documentation.
- Never add an unverified `[tool.uv]` key. An invalid key can make uv discard the entire table,
  including the TensorRT pin and cross-platform environment constraints.
- LiteRT support is paused until `litert-torch` supports the current PyTorch range.

## Data and training

- Class IDs and `label_to_name` must be zero-based and contiguous.
- Instance segmentation requires polygon annotations; bbox-only data must fail.
- Four-channel input uses RGB+extra `.npy` arrays. Do not add TIFF support; OpenCV corrupts some
  four-channel TIFFs. `train.in_channels` supports only 3 or 4.
- Mask tasks need a `dfine_seg_*` checkpoint to restore the trained MaskDecoder. There are no released
  semantic-segmentation weights; semantic segmentation fine-tunes from instance-segmentation weights.
- `model.pt` is the best checkpoint and is used by validation, inference, export, and bench. `last.pt`
  is the latest state and is not resumable because checkpoints contain no optimizer or EMA state.

## Public API and checkpoints

- `import dfine_seg` must remain torch-only. Do not pull Hydra, WandB, Albumentations, Matplotlib,
  pandas, sklearn, or torchmetrics into API module scope.
- `load_model` is a factory returning the concrete backend model. Do not wrap it or force outputs to
  CPU. Graph artifacts carry their task; `task=` is forwarded only for `.pt` files.
- Keep `Visualizer(model)` as the shared drawing entry point. It accepts three-channel BGR; callers
  must prepare `.npy` RGB(+extra) inputs before drawing.
- Standalone modules under `dfine_seg/infer/` are intentionally self-contained because users copy
  them independently. Do not consolidate their shared-looking code.
- Save checkpoints with `save_checkpoint` and read them only through `unwrap_checkpoint`. Metadata
  must contain plain Python values compatible with `torch.load(..., weights_only=True)`.
- Infer architecture from weight shapes, not checkpoint metadata. Preprocessing precedence is:
  explicit arguments → checkpoint metadata → sidecar config → defaults. Hydra commands pass explicit
  preprocessing values, so the live config wins.

## Export constraints

- TensorRT 10.13.3.9 batched engines are slot-dependent. Run exported TensorRT engines at batch 1.
- Do not retry the rejected segment TensorRT graph fusions (mask-feature output, fp16 mask output,
  in-graph NMS). The measured wins are already in the client wrappers.
- Semantic segmentation intentionally uses argmax followed by nearest-neighbor resize. Upsampling
  logits before argmax improved mIoU only slightly and increased TensorRT latency substantially.

## Verification

- Normal changes: `uv run ruff format . && uv run ruff check . && make test-fast`.
- Model-output changes: also run `make test`, regenerate fixtures with
  `uv run python -m tests.generate_fixtures` and review the updated baseline.

## Working rules

- Prefer the smallest clear implementation and short comments that explain only non-obvious choices.
- Do not commit, push, create branches, or open pull requests unless explicitly requested. Leave
  changes uncommitted for review.
