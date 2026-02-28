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
