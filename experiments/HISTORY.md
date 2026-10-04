# What we tried before (2026-06 → 2026-09)

Full logs are in git history (`git log -- experiments/lab_notebook.md`, removed 2026-09-29).

## Detection campaign (VisDrone, S, ImageNet-init, 2 seeds × 60 min)
- **Won and shipped:** Muon on enc/dec attention and MLP matrices (+0.004 mAP), then Adan on the non-Muon
  groups with aux LR ×5 (+0.005 mAP). Both also held on full 75-epoch COCO-init runs at X.
- **Rejected (tie or worse):** MAL loss, 300 CDN queries, dense O2O (mosaic 1.0), PMC and HMC
  matcher costs, Cautious AdamW, Moonlight RMS matching, Muon weight decay (0.1 and 0.03),
  IA-BCE targets, higher backbone LR, PreciseBN, SPD-Conv, RMSNorm+SwiGLU decoder.
- **Shelved:** QK-norm fixes the fp16 NaN but TensorRT mis-executes the trained checkpoint. The NaN was
  fixed instead by bf16 AMP.
- **Lessons:** the optimizer was the only axis that moved. Walltime caps plus a long `epochs` horizon
  leave the LR un-annealed, so screens must finish their schedule. Always judge the TensorRT artifact,
  because torch can look fine while the engine is broken.

## Semantic segmentation (Cityscapes, S @640, COCO seg init)
- Dense-mask mosaic (affine window-crop, 0.5) was the big win. S reaches 0.728 mIoU at 2.0 ms TRT
  fp16 after 75 epochs.
- Rejected: bilinear logits upsample before argmax (+0.004 mIoU for about 20% more latency) and six
  TRT engine-side latency experiments. The sem_seg TRT path is at the hardware floor: resize is
  memcpy-bound, H2D is PCIe-bound and the head conv runs at 96% of fp16 peak.
- The gap to EfficientNet-B5 was in the recipe and regularization, not in resolution.

## Standing facts (sem_seg recipe, measured 2026-09-30)
- EMA on the 40-epoch Cityscapes screen ends at momentum 0.976, a 41-iteration window; the 30-epoch GOOSE
  run reaches 2.4 epochs. The two presets get different EMA behaviour.
- Cityscapes (2048×1024) and GOOSE (2048×1000) are both 2:1 landscape, squished 3.2× horizontally and
  1.6× vertically into 640×640. Past resolution runs were square (S) except the July M runs at 640×1280 (2:1, trained on the pre-fix pos-embed code).
- Cityscapes rare classes: bus and train appear in 8 % of train images, truck 11 %, motorcycle 20 %;
  rider, motorcycle, traffic-light and truck each hold under 0.25 % of pixels.
- Photometric augmentation is effectively off (all probabilities 0.01–0.02); mosaic is the only real
  regulariser.
- The S seg head is about 30 GMAC (up_conv 15.1, neck 11.3, fusion 3.8), the yardstick for head changes.
