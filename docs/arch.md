# Architecture

D-FINE-seg is one network with three parts: an HGNetv2 **backbone**, D-FINE's **HybridEncoder**, and a task-specific **decoder**. Backbone and encoder are the same for every task; the task only decides which decoder runs on top.

```
image [B, 3|4, H, W]
 └─ HGNetv2 backbone ──────────────> C3, C4, C5   (strides 8, 16, 32)
     └─ HybridEncoder (AIFI + FPN/PAN) ──> P3, P4, P5   (hidden_dim channels each)
         ├─ detect:  D-FINE decoder ─────────────────────> boxes, scores
         ├─ segment: D-FINE decoder + MaskDecoder ───────> + one mask per box
         ├─ kpts:    D-FINE decoder + keypoint head/decoder ─> + K points per box
         └─ sem_seg: SemSegDecoder (no queries) ─────────> class per pixel
```

Shapes below are for size S at 640x640 unless stated. Code: `dfine_seg/model/arch/` (`hgnetv2.py`, `hybrid_encoder.py`, `dfine_decoder.py`), sizes in `dfine_seg/model/configs.py`.

## Backbone: HGNetv2

- A CNN (PP-HGNetV2: B0 for N/S, B2 for M, B4 for L, B5 for X), ImageNet-pretrained at the start of D-FINE's COCO training; its weights come with the detection checkpoints.
- Stem: two stride-2 convs with a small max-pool / conv branch in between, output at stride 4.
- 4 stages at strides 4, 8, 16, 32; stages 2-4 start with a depthwise stride-2 conv. Each stage stacks HG blocks: a chain of convs (3x3, or 1x1 + depthwise 5x5 "light" convs in stages 3-4) whose outputs are all concatenated with the block input, then squeezed by two 1x1 convs (to half the output channels, then to the output).
- Returns stages 2-4 (strides 8/16/32; 256/512/1024 channels for B0). N returns only strides 16/32.
- Small backbones (B0/B2) add a learnable scale + bias after activations; L/X use frozen BatchNorm.
- 4-channel input (RGB + one extra modality): the stem's first conv gets a 4th input channel initialized to the mean of the RGB filters.

## Encoder: HybridEncoder

- 1x1 conv + BN projects each backbone level to `hidden_dim` (128 N, 256 S/M/L, 384 X).
- **AIFI**: one transformer encoder layer (8 heads, GELU FFN) over the stride-32 map only (20x20 = 400 tokens), with a 2D sin-cos position embedding. Global context is added where it is cheap.
- **CCFF**: a top-down FPN (1x1 lateral conv, nearest x2 upsample, concat, fusion block) followed by a bottom-up PAN (1x1 conv + depthwise stride-2 conv, concat, fusion block). The fusion block is RepNCSPELAN4: 1x1 conv, split channels, two chained CSP stages of RepVGG blocks (parallel 3x3 + 1x1 convs that fold into one 3x3 conv at export), each followed by a 3x3 conv, concat of all four parts, 1x1 conv.
- Output: P3, P4, P5 at strides 8/16/32 (80x80, 40x40, 20x20), all `hidden_dim` channels. N outputs only strides 16/32.

## Detection decoder (D-FINE)

Used by `detect`, `segment` and `kpts`.

- **Memory**: P3-P5 are flattened into one token sequence (80² + 40² + 20² = 8400 tokens for S at 640).
- **Query selection**: every memory token has an anchor box (its cell center, size 0.05 x 2^level). A Linear + LayerNorm and a class head score each token; the top 300 become queries. A query's content is its projected memory feature and its reference box is an MLP's correction of the anchor.
- **Decoder layers** (3 for N/S, 4 M, 6 L/X), each:
  - self-attention among the 300 queries, with a position embedding computed by an MLP from the reference box;
  - multi-scale deformable cross-attention: each query samples a few points per head and level (3/6/3 at strides 8/16/32 for S) around its box center, with offsets scaled by the box size, and takes their weighted sum;
  - a gate (two learned sigmoid gates weight the query and the attention output before they are summed, then LayerNorm) in place of the residual add, then an FFN.
- **Boxes, FDR (Fine-grained Distribution Refinement)**: after layer 1 an MLP refines the reference into an initial box, which stays fixed. Every layer's box head then predicts, for each of the 4 edges, a distribution over 33 bins of a non-uniform weighting function (fine steps near 0, coarse at the ends). Its expectation is how far the edge moves from the initial box's edge, in units of box size / `reg_scale` (1/4 of the box for N-L, 1/8 for X; at most one box size either way). Each layer adds its logits to the previous layer's, so layers refine distributions rather than regress fresh coordinates; the next layer's reference is the current box.
- **Scores**: a Linear class head per layer, plus LQE: an MLP over the top-4 probabilities (and their mean) of each edge distribution adds a localization-quality term to the class logits (a sharp distribution means a confident box).
- **Training**:
  - one-to-one Hungarian matching (focal class cost, L1 and GIoU box costs);
  - VFL classification, L1 + GIoU boxes, FGL (pushes each edge distribution onto the two bins around the true distance) and DDF (the last layer's distributions distill into the earlier ones);
  - losses on every layer, the encoder's top-300 proposals and the initial boxes;
  - contrastive denoising: extra queries built from noised GT boxes and labels, in 100 // (most GTs in any image of the batch) groups. Each group holds a positive copy of every GT (reconstruct it) and a more heavily noised negative copy (reject it). An attention mask hides the denoising queries from the real ones and the groups from each other.
- **Inference**: every layer's box head still runs (each refines the previous layer's distributions), but the class head and LQE run only on the last layer; export fuses the RepVGG branches. Scores are sigmoids per class; the top 300 (query, class) pairs above `conf_thresh` are kept. One-to-one matching makes duplicates rare; the inference wrappers still apply NMS at IoU 0.7 by default (`apply_nms`).

## Instance segmentation (`task: segment`)

- **MaskDecoder** (pixel features): per PAN level a 1x1 conv + GroupNorm. Coarser levels are bilinearly upsampled to stride 8 and summed, then a 3x3 conv + GN + ReLU, bilinear x2 and another 3x3 conv + GN + ReLU. Output: mask features `[B, 256, H/4, W/4]` (160x160). N has no stride-8 encoder level, so it adds the backbone's stride-8 feature as an extra input.
- **Mask head** (per instance): a 3-layer MLP turns each final decoder query into a 256-d mask embedding. Its dot product with the mask features (scaled by 1/sqrt(256)) gives that query's mask logits at 1/4 resolution.
- **Training**: matching adds focal and dice mask costs. Losses are BCE + dice computed only inside the GT box (normalized by its area), on every decoder layer and on denoising queries.
- **Inference**: masks only for the kept detections: sigmoid, cut the letterbox padding, one bilinear resize to the original image, threshold, clear pixels outside the box.
- Adds ~1.6M parameters to S (MaskDecoder 1.38M, mask head 0.2M).

## Semantic segmentation (`task: sem_seg`)

- Replaces the whole D-FINE decoder: no queries, matching or boxes, just a dense per-pixel classifier on the PAN outputs.
- The same MaskDecoder fuses P3-P5 to 1/4 resolution. Its weights can load from the `segment` checkpoints (`dfine init --task sem_seg` sets that init); from the default detection weights it starts random.
- Neck: two 3x3 conv + GN + ReLU (128 channels), dropout, then a 1x1 classifier gives class logits at 1/4, bilinearly upsampled x4 to the input size.
- Training: cross-entropy + multi-class dice, plus a train-only auxiliary classifier on the stride-8 PAN feature (CE, weight 0.4).
- Inference: argmax at input resolution, then a nearest-neighbor resize of the label map to the original image. Exported graphs end in that argmax and output an int32 label map.
- Decoder 2.1M parameters; S totals 8.0M (vs 10.3M for detection).

## Keypoints (`task: kpts`)

Every detection also gets K points, each with (x, y, presence logit, visible logit). Two stages:

- **Stage 1, per-query head**: after each box-decoder layer, a 3-layer MLP on each query outputs K x (dx, dy, presence logit, visible logit). A point is the box center plus the offset x box size, and the offset is not bounded to the box. Each layer corrects the previous layer's offset, like D-FINE's box refinement. Zero-initialized, so points start at the box center.
- **Stage 2, keypoint decoder** for the top 50 queries by class score. Each point becomes a token: the query's embedding projected, plus a learned "which point am I" embedding. Each of the 2 layers does:
  - self-attention among the K tokens of one object, so points see each other;
  - deformable cross-attention from each token to the encoder memory (the same multi-scale features the box decoder samples), around that point's current position, with offsets scaled by 0.25 x the box size;
  - a gate and FFN, then zero-initialized heads that predict a residual (dx, dy) x box size and residual logits;
  - layer 2 samples around layer 1's refined points.

  Queries outside the top 50 keep their stage-1 points.
- **Relation to D-FINE**: it's the same mechanism as D-FINE's decoder (it reuses its decoder layer), applied per point instead of per object. The cross-attention is point token → image features, not point → box.
- **Quality head and rescoring**: an MLP over the mean of an object's K tokens predicts its OKS (trained toward the OKS it achieved). At inference, refined detections score `cls^0.5 x quality^0.5`, so a detection whose points are likely off ranks lower.
- **Keypoint denoising** (train-only): the keypoint decoder also runs on the positive denoising queries of one group, starting from GT points plus uniform noise of up to 0.1 x box size, and learns to pull them back.
- **Training**: matching adds a point L1 cost and a 1 - OKS cost. Losses are box-normalized L1, 1 - OKS, presence BCE (labeled vs not, all K slots) and visible BCE (`v == 2` vs `v == 1`, labeled points only), plus the quality loss. They apply to every box-decoder layer's stage-1 points, each keypoint-decoder layer and the denoising queries.
- **Output**: `kpts [N, K, 4]` per detection: x, y in original-image pixels, presence and visible scores (sigmoids). All K points are returned; thresholding presence is up to the caller.
- **TensorRT**: the cross-attention is split into at least 2 chunks and every GridSample is pinned to fp32 at export, working around two TensorRT bugs (wrong results for chained GridSamples sharing an input; fp16 mis-execution of the keypoint decoder).
- Adds ~3.2M parameters to S (stage-1 heads 0.45M, keypoint decoder 2.79M; 13.5M total with one class).

## Model sizes

Parameters for detection with 80 classes.

| size | backbone | encoder levels (strides) | hidden dim (enc / dec) | decoder layers | points per level | params (M) backbone / encoder / decoder / total |
|---|---|---|---|---|---|---|
| N | B0 | 16, 32 | 128 / 128 | 3 | 6, 6 | 1.85 / 0.70 / 1.23 / 3.78 |
| S | B0 | 8, 16, 32 | 256 / 256 | 3 | 3, 6, 3 | 1.85 / 4.02 / 4.45 / 10.32 |
| M | B2 | 8, 16, 32 | 256 / 256 | 4 | 3, 6, 3 | 6.03 / 7.80 / 5.76 / 19.59 |
| L | B4 | 8, 16, 32 | 256 / 256 | 6 | 3, 6, 3 | 13.51 / 9.34 / 8.39 / 31.24 |
| X | B5 | 8, 16, 32 | 384 / 256 | 6 | 3, 6, 3 | 33.23 / 20.70 / 8.69 / 62.62 |
