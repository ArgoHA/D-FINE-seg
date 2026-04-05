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
