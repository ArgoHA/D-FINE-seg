# Changelog

All notable changes to D-FINE-seg since the paper release will be documented in this file.

## 2026-02-28 - Improve Nano segmentation quality

- **Nano mask output resolution: 1/8 -> 1/4.** The backbone's 1/8 feature (HGNetV2 stage 2) is now passed directly to MaskDecoder, bypassing HybridEncoder. Previously, Nano only used 2 PAN scales (1/16, 1/32), producing 1/8 mask output — coarser than the 1/4 output of S/M/L/X models which use 3 scales (1/8, 1/16, 1/32). The low-level feature is extracted before the encoder and routed straight to MaskDecoder, keeping encoder computation unchanged.
- **Nano `mask_dim` reduced from 256 to 128**, matching the encoder hidden dimension for better efficiency.

#### Results (TACO dataset)

| Metric | Before | After |
|--------|--------|-------|
| mIoU   | 0.096  | 0.107 (+11% relative) |
| Latency | 4.0 ms | 4.1 ms (+2%) |

## 2026-03-05 - Implement CoreML export and inference

- Export now also supports CoreML in fp32 and fp16.
- New inference module for CoreML. On m1pro fp32 was faster, so it is used by default
- Readme updated with benchmarks (TACO detectoin and segmentation, S model, m1 pro model)

## 2026-03-11 - CoreML int8

- Add int8 quantzation for CoreML, ruexported by default alongside with fp32 versionduring `make export`
- Adepted `make bench` to supprot macos and linux platforms automatically. Torch, OpenVINO, ONNX run for both. TensorRT - linux, CoreML - macos.

## 2026-04-05 - LiteRT export and COCO segmentation pretrained weights

- Add LiteRT export, inference class and update bench.py to include LiteRT
- Add support to coco dataset formats
- Add pretrained weights on COCO dataset for segmentation models (n, s, m, l, x)
- Convert all pretrained models to this repo format and pth -> pt

## 2026-04-14 - Run NMS in inference classes by default

Although D-FINE doesn't require a NMS, it still helps to boost the accuracy with a tiny latency increase. TensorRT FP16, 5070ti, model D-FINEm, VisDrone dataset:

| Metric | F1-score | Latency |
|--------|--------|-------|
| No NMS | 0.587 | 3.6 ms |
| With NMS | 0.605 | 3.8 ms |

Same behaviour on TACO dataset for both detectin and segmentation models.

## 2026-05-01 - Optimize TensorRT inference class

Several improvements in the TensorRT inference class. Although it doesn't support dynamic input size, it is very well optimized for the static input. With S size model latency went from 3.1ms to 2.1ms without changes in the accuracy.

Minor improvement - now pretrained weigts automatically download from HuggingFace

## 2026-05-24 - Multi-channel input support (RGB + thermal / depth / NIR / ...)

- New `train.in_channels` config (default 3). Set to `4` to train on RGB + one extra modality (thermal / depth / NIR). Supported range is 3 or 4 — higher counts hit cv2 Scalar / Albumentations limits and are rejected at config load.
- Multi-channel images are stored as `.npy` (HWC uint8) — byte-faithful via `np.load`, unlike multi-channel TIFF which `cv2.imread` silently mangles. Channel convention: RGB in planes 0..2, extras in 3..N-1.
- HGNetv2 stem conv is rewired for `in_channels=4`. Pretrained 3-channel weights are reused: stem is inflated to 4 channels by tiling the RGB filter mean, so COCO-pretrained fine-tuning still works out of the box.
- All inference backends (torch, onnx, openvino, tensorrt, coreml, litert) auto-detect channel count from the exported model and preprocess accordingly.
- `dfine_seg/etl/m3fd_to_yolo.py` converts the [M3FD](https://github.com/JinyuanLiu-CV/TarDAL) RGB+thermal benchmark (VOC XML + Vis/Ir PNGs) into the new layout as a reference example.

## 2026-06-21 - Muon + Adan optimizers

- New `dfine_seg/model/muon.py` (`MuonWithAuxAdam`): Muon (Jordan et al., 2024) routes the encoder/decoder attention/MLP weight matrices to Newton-Schulz-orthogonalized momentum, while backbone, norms, biases, embeddings, and det/mask heads stay on an AdamW aux path inside one optimizer. **On by default** (`train.use_muon: True`).
- Adan (Xie et al., arXiv:2208.06677) is selectable for the aux groups via `train.aux_optimizer: adamw|adan`, with `train.adan_lr_mult` and `train.adan_betas` knobs.
- Muon + Adan is the new best recipe on the VisDrone screen: +0.0078 mAP_50_95 over the AdamW-X reference, latency-neutral.

## 2026-07-11 - Semantic segmentation task

- New third task `task: sem_seg` — dense per-pixel classification, full pipeline (train / infer / export / bench).
- Head: `SemSegDecoder` reuses the pretrained `MaskDecoder` fuser from `dfine_seg_<size>_coco.pt` (backbone + encoder + fuser transfer; only the small neck/classifier train from scratch) + a train-only aux head for deep supervision. No queries, no NMS. Loss: CE + multi-class Dice + 0.4 aux-CE.
- Data: `labels/<stem>.png` (single-channel uint8, pixel value = class id); `train.sem_seg.ignore_index` (default 255) excluded from loss/metrics and used as mask fill for pad-introducing augs.
- Eval: decision metric mIoU from a pixel confusion matrix at original image resolution (same protocol in training eval and bench); new metrics — mIoU (macro) + pixel_acc (micro).
- Export: one fused-argmax graph for every backend — single int32 `sem_seg` `[B, H, W]` output; parity check compares per-pixel argmax agreement. Wrappers return `out["sem_seg"]` `[H, W]` label map at original resolution.

#### Results (Cityscapes Dataset, S @ 640, RTX 5070 Ti)

| Backend | mIoU | Latency |
|--------|------|---------|
| PyTorch fp32 | 0.728 | 9.8 ms |
| TensorRT fp16 | 0.728 | 2.0 ms |

## 2026-07-28 - Instance segmentation postprocessing speedup

E2E inference latency of instance segmentation model improved by 25% (4.1ms -> 3.09ms) for TensorRT. Minor speedups for other formats too.

## 2026-08-17 - Pypi

D-FINE-seg is now pip-installable, has a public Python API and CLI. Repo folders were renamed, inference class names were standardized (PEP8). Torch model now autodetects `model_name`, `task`, `num_classes` and `in_channels`. Demo is also updated, now has a dropdown menu with 10 pretrained weights and allows user to load his weights from the UI. Added meta data to the checkpoints, so pt weights contain training info like the model_name, task, num_classes, label_to_name...

## 2026-08-24 - Visualizer in the public API

Python api now supports model output visualization

## 2026-09-16 - v0.4.2

- Add a standalone C++ TensorRT inference library for detection, instance segmentation, and semantic segmentation, plus an end-to-end NVDEC -> inference -> GPU overlay -> NVENC video pipeline.
- Add box-only SAM3 inference with `task="detect"`, avoiding full-resolution mask materialization when masks are not needed.
- Add box attention heatmap visualization tooling.
- Increase the distributed-training timeout and simplify training, inference, validation, and error-analysis internals with expanded regression coverage.
- Add machine-readable citation metadata and PyPI links for the paper, model weights, documentation, and changelog.

## 2026-09-30 - Fix position embedding for non-square input

The HybridEncoder sin-cos position embedding built its grid in (w, h) order while tokens are flattened row-major, so with a non-square img_size most tokens got the wrong position (the bug is inherited from RT-DETR/D-FINE). The embedding is now built in (h, w) order. Square inputs are identical to before, so existing weights, exported models and 640×640 results are unaffected; trained models even with non-square img_size were able to adapt in my experiments, the only time I saw accuracy degradation (2%) was when I took a trained model on square and inferenced on non-sqaure:

640x640 (as trained) ->  0.595
448x896 (with bug) -> 0.513
448x896 (with fix) -> 0.520

Note: Checkpoints trained at a non-square `img_size` before this fix learned the old layout and lose some accuracy under the new code; retrain them, or keep them on the previous version.

## 2026-10-01 - Remove box rounding

Minor difference, previously box conversion floored x1/y1 and ceiled x2/y2 at the model input resolution before scaling to the original image, which snapped every box edge outward.
On Cityscapes (S, 640×640, 3 seeds) removing it gave +0.004–0.008 mAP50 and +0.009–0.015 mAP50-95, mostly on small objects; rounding the training labels made no difference.
Validation used to round the GT the same way, which inflated it, so reported val mAP is now about 0.005–0.010 lower for the same model; this is a measurement change, not a regression.
Re-export models to get the unrounded boxes from ONNX/TensorRT/OpenVINO/CoreML.

## 2026-10-04 - Rare class sampling

LVIS repeat-factor sampling: draw train images with rare classes more often (not under DDP). Turned off by default. Dataset-dependent: +0.004 mIoU on GOOSE, flat on Cityscapes sem_seg.

## 2026-10-10 - Knowledge distillation

Train a bigger dfine-seg model and use it as a teacher for a smaller dfine-seg model. Detection, instance semgentation and semantic segmentation are supported. ~15% longer training for ~1% higher metrics. Detailed results in experiments/kd_exps.csv

## 2026-10-15 - Cityscapes semantic segmentation weights

- Five released Cityscapes sem-seg checkpoints, one per size: `dfine_sem_{n,s,m,l,x}_cityscapes.pt`. Cityscapes val, 1024x2048 whole-image, TensorRT 10.13.3.9 fp16, batch 1, RTX 5070 Ti - mIoU 77.99 (N), 81.12 (S), 81.41 (M), 83.80 (L), 84.84 (X). N was initialized from COCO detection weights (no obj2coco N checkpoint exists), S/M/L/X from obj2coco (Objects365 pretraining, then COCO). Checkpoints are best-on-val epochs; last-epoch numbers and the full 28-model comparison live in [cityscapes-benchmark](https://github.com/ArgoHA/cityscapes-benchmark).
- `load_model("x", task="sem_seg")` auto-downloads them (`dataset` defaults to cityscapes for this task); class names and the 1024x2048 input size come from the checkpoint metadata.
- Fine-tune a custom sem-seg dataset from these weights with a different class count: `train.pretrained_model_path: dfine_sem_l_cityscapes.pt`. The classifier and aux head re-initialize for the new class set; backbone, encoder, mask fuser and seg neck load.

## 2026-10-18 - Instance segmentation denoising loss fix

The last decoder layer's denoising queries now get class/box supervision in instance segmentation (previously they were silently dropped, so only their mask loss was applied). Cityscapes regression: mAP_50_mask 0.496 -> 0.507 (can be noise), f1 unchanged. Aux mask logits are no longer computed at eval.

## 2026-10-29 - Adding Keypoint estimation (`task: kpts`)

- New task: every detection also predicts K points (YOLO-pose labels `class xc yc w h x1 y1 v1 ...` or COCO `keypoints`, `train.kpts` config block). A small MLP per decoder layer predicts initial box-relative points, plus a presence and a visibility score per point. A 2-layer keypoint decoder then refines the top 50 detections: one token per point, self-attention between the points of an object and deformable cross-attention around each point, trained with keypoint denoising. A keypoint-quality head (predicted OKS) rescores the refined detections: `score = cls^0.5 * quality^0.5`. Losses: L1 + OKS + presence/visibility BCE + quality BCE; Hungarian cost includes keypoints. Flips (with `flip_idx`), mosaic, affine and letterbox carry the points; coarse dropout marks the points it hides as occluded.
- Metrics: `f1_kpt` (OKS >= 0.5), COCO keypoint `mAP_50_95_kpt`, `kpt_px_err`, `presence_f1`, `visible_acc` (more in `extended_metrics.csv`); best checkpoint = mean of `mAP_50_95_kpt` and box `mAP_50_95`.
- Export: one extra `kpts` output `[B, K_top, num_kpts, 4]` for ONNX, TensorRT, OpenVINO and CoreML; all Python wrappers return `kpts` `[N, K, 4]` in original image pixels, `dfine infer` writes YOLO-pose lines, `Visualizer` draws points and skeleton. Not yet in the C++ library or the int8 export scripts. TensorRT engines for `kpts` keep GridSample in fp32 (TensorRT 10.13 mis-executes the keypoint decoder's chained GridSamples in fp16).
- `train.kpts.freeze_except_kpt` trains only the keypoint modules (per-query head and keypoint decoder).

#### Results (S, 640x640, init from COCO person detection weights, RTX 5070 Ti, batch 1)

| Task | Metric | D-FINE-seg S | Reference |
|------|--------|--------------|-----------|
| 17-point COCO pose, val2017, 20 epochs | keypoint AP / person box AP / TensorRT fp16 latency | 63.0 / 63.6 / 2.10 ms | YOLO26s-pose, same 20 epochs from COCO detection weights: 55.0 / 59.8 / 2.13 ms; released YOLO26s-pose (300 epochs, Objects365 init): 62.7 / - / 2.14 ms |
| One ground-contact point (ankle midpoint), COCO val, 10 epochs | median error / box height | 0.024 | 0.123 (box bottom-center) |

Keypoint AP is the official COCO scorer on all val2017 images with people; latency is per-image predict at conf 0.5 through each framework's own pipeline. TensorRT matches PyTorch (63.0 vs 63.0 AP). The previous per-detection MLP head alone reached 37.8 AP. The single predicted point beats the box bottom-center on 93% of people (97% when the ankles are visible).

## 2026-10-31 - Semantic segmentation probability output

`export.sem_seg_probs: True` adds a second output to the sem_seg graph: the per-class softmax at 1/4 of the model input, next to the unchanged int32 label map. Every wrapper then also returns `sem_seg_probs` `[C, h, w]` and `sem_seg_probs_tf` (3x3 original-pixel -> probs-pixel transform, exact under letterbox; `K' = tf @ K` for camera projection). For `.pt`: `TorchModel(..., sem_seg_probs=True)`. Intended for consumers that fuse per-class confidence (traversability, lidar painting, BEV). The C++ core accepts these engines and ignores the extra output. Measured on TensorRT fp16 (RTX 5070 Ti, Cityscapes val at 1024x2048): the engine gets ~0.02 ms slower and end-to-end latency stays within run-to-run noise (S 6.66 -> 6.69 ms, X 14.87 -> 14.77-14.98 ms); the label map and mIoU are unchanged.
