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
