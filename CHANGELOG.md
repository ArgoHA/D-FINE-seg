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
- `src/etl/m3fd_to_yolo.py` converts the [M3FD](https://github.com/JinyuanLiu-CV/TarDAL) RGB+thermal benchmark (VOC XML + Vis/Ir PNGs) into the new layout as a reference example.

## 2026-06-21 - Muon + Adan optimizers

- New `src/d_fine/muon.py` (`MuonWithAuxAdam`): Muon (Jordan et al., 2024) routes the encoder/decoder attention/MLP weight matrices to Newton-Schulz-orthogonalized momentum, while backbone, norms, biases, embeddings, and det/mask heads stay on an AdamW aux path inside one optimizer. **On by default** (`train.use_muon: True`).
- Adan (Xie et al., arXiv:2208.06677) is selectable for the aux groups via `train.aux_optimizer: adamw|adan`, with `train.adan_lr_mult` and `train.adan_betas` knobs.
- Muon + Adan is the new best recipe on the VisDrone screen: +0.0078 mAP_50_95 over the AdamW-X reference, latency-neutral.

## 2026-07-06 - Semantic segmentation task

- New third task `task: sem_seg` — dense per-pixel classification, full pipeline (train / infer / export / bench).
- Head: `SemSegDecoder` reuses the pretrained `MaskDecoder` fuser from `dfine_seg_<size>_coco.pt` (backbone + encoder + fuser transfer; only the small neck/classifier train from scratch) + a train-only aux head for deep supervision. No queries, no NMS. Loss: CE + multi-class Dice + 0.4 aux-CE.
- Data: `masks/<stem>.png` (single-channel uint8, pixel value = class id); `train.sem_seg.ignore_index` (default 255) excluded from loss/metrics and used as mask fill for pad-introducing augs.
- Eval: decision metric mIoU from a pixel confusion matrix at original image resolution (same protocol in training eval and bench); new metrics — mIoU (macro) + pixel_acc (micro).
- Export: one fused-argmax graph for every backend — single int32 `sem_seg` `[B, H, W]` output; parity check compares per-pixel argmax agreement. Wrappers return `out["sem_seg"]` `[H, W]` label map at original resolution.

#### Results (Semantic Drone Dataset, S @ 640, 23 classes, RTX 5070 Ti)

| Backend | mIoU | Latency |
|--------|------|---------|
| PyTorch fp32 | 0.644 | 10.3 ms |
| TensorRT fp16 | 0.642 | 2.9 ms |
