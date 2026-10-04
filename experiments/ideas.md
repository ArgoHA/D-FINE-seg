# Ideas queue

The research agent appends ideas and the executor updates their status. Statuses: `pending` → ✅ kept /
❌ rejected / 💥 failed. Keep each entry short.

<!-- template
## <slug> — pending
- Change: … (files)
- Why: … (source)
- Latency risk: none / low / high
- Result: ΔmIoU …, latency … ms → verdict, one-line reason
-->

# Batch 1 — 2026-09-30

Baseline `exp_base` cd5c783: TRT mIoU 0.729 / 0.732 (mean 0.7305, spread 0.003) @ 1.87 ms; off-road
0.625 / 0.623. Keep bar: ΔmIoU ≥ +0.004 at ≤ 1.93 ms. Ranked by expected gain ÷ implementation cost.
Data and recipe facts behind the ranking are in `HISTORY.md` → Standing facts.

## ohem-ce — ❌
- Change: online hard-example mining for `loss_ce` and `loss_aux`: per-pixel CE (`reduction="none"`), keep
  pixels whose GT-class prob < 0.7, but at least `valid/16` hardest; Dice untouched. Knob
  `train.sem_seg.ohem_thresh: 0.7` (null = off) in `config.yaml` + `default.yaml`, plumbed through
  `build_loss` (`dfine_seg/model/dfine.py`) into `dfine_seg/model/sem_seg_criterion.py`. Test in
  `tests/unit/test_sem_seg.py`.
- Why: 37 % of pixels are road, 24 % building; plain CE is dominated by easy interior pixels after a few
  epochs. OHEM is the default CE in every real-time Cityscapes recipe (BiSeNet/STDC thresh 0.7 n_min=1/16,
  DDRNet/PIDNet thresh 0.9). Source: Shrivastava et al., CVPR 2016 (arXiv:1604.03540); STDC (CVPR 2021)
  code. Not in HISTORY (never tried on sem_seg). sem_seg-only code.
- Latency risk: none (train-only).
- Result: seed 42 TRT 0.723 (torch 0.7247) vs baseline s42 0.729/0.7288 → ΔmIoU −0.006, 1.84 ms; seed 123 skipped. ❌ clear loss on both metrics; likely redundant with Dice, which already rebalances pixels (branch exp/ohem-ce 3867a85).

## native-aspect-input — ✅
- Change: train/export at the dataset's native aspect ratio at the same pixel budget: `train.native_aspect:
  null` (task default: true for sem_seg, false otherwise, same pattern as `mosaic_prob`). When on, resolve
  `img_size` once (where the config is loaded, so train/export/bench agree): median (H, W) of the train
  split from image headers (PIL `.size`, no decode), scale to `h*w ≈ img_size[0]*img_size[1]`, snap both
  to /32 → 448×896 for Cityscapes (401 k px, −2 %), 448×928 for GOOSE (+1.5 %). Files:
  `dfine_seg/config/resolve.py` (or the Hydra entry in `dfine_seg/dl/utils.py`), `config.yaml`,
  `default.yaml`, `tests/unit/test_config_template.py`. Everything downstream already takes (h, w):
  `A.Resize(target_h, target_w)`, 2H×2W mosaic canvas, `build_2d_sincos_position_embedding(w, h)`,
  `SemSegExportWrapper`, TRT wrapper `input_width/input_height`, ckpt meta `img_size`.
- Why: 640×640 squishes 2:1 images 3.2× horizontally but 1.6× vertically; poles, people, signs and
  lights lose most of their width, and those thin classes are where the S deficit sits (July upsample A/B:
  gains concentrated in pole/sign/light). At 448×896 both axes shrink 2.29×, shapes match the COCO
  pretraining statistics, and FLOPs are unchanged. All prior resolution runs (`sem_seg_m_896`, `_960`,
  `_1280`, letterbox) were square; a non-square input was never tried. General: any dataset with a
  consistent aspect ratio benefits, and GOOSE is also 2:1.
- Latency risk: low. Same pixel count, same H2D bytes; AIFI sees 14×28 = 392 tokens vs 400. Verify the
  engine at 448×896 is within 1.03× (tile efficiency can differ slightly).
- Result: img_size [448, 896] (plain config, no auto-resolve knob) + AIFI pos-embed fix (grid was (w, h), scrambled non-square; square bit-identical). TRT 0.736 / 0.742 (mean 0.739, torch 0.7353 / 0.7428) vs 0.7305 → ΔmIoU +0.0085 @ 1.87 / 1.86 ms (=). ✅ ff-merged a12eb96. Note: root img_size default now non-square for all tasks; scope before shipping.

## rare-class-sampling — ❌ superseded (RCS removed 2026-10-04; RFS kept opt-in, see Follow-ups)
- Change: class-balanced image sampling for sem_seg train loaders. One-time scan of `labels/*.png` for the
  per-image class set (cache `<data_path>/class_presence.json`); DAFormer RCS weights
  `P(c) ∝ exp((1 − f_c)/T)`, T = 0.01, with `f_c` = share of images containing c; image weight
  `w_i = Σ_{c∈i} P(c)/n_c`; `WeightedRandomSampler(w, num_samples=len(ds), replacement=True)` in
  `Loader._build_dataloader_impl` (`dfine_seg/dl/dataset.py`), only for `task == sem_seg`, mode train,
  not distributed (DDP keeps `DistributedSampler`). Knob `train.sem_seg.rare_class_sampling: true`.
- Why: bus/train appear in 8 % of images, truck 11 %, motorcycle 20 %; with 40 epochs these classes get
  ~3 effective epochs. Dice already balances pixels within a batch but cannot help when the class is
  absent from the batch. Class-uniform sampling gave +1.1 mIoU on Cityscapes (Zhu et al., CVPR 2019,
  arXiv:1812.01593 §4); RCS gave +2.0 in DAFormer (Hoyer et al., CVPR 2022, arXiv:2111.14887). Transfers
  to off-road, where animal/human/water/sign are the rare classes. Scale-jitter's July loss came from
  starving rare classes; this is the opposite lever. Flag: `Loader` is shared, change is task-guarded.
- Latency risk: none.
- Result: f_c = pixel frequency (DAFormer; image share at T=0.01 collapsed onto 'train', 99 % of draws). Rare-class image share ~2× (train 5→14 %, bus 9→20 %). TRT 0.742 / 0.744 (mean 0.743, torch 0.7433 / 0.745) vs 0.739 → ΔmIoU +0.004 @ 1.87 / 1.88 ms. ❌ below the +0.006 bar (baseline spread); +73 lines of loader code. Seeds tighter (spread 0.002). Candidate to retry stacked on a later exp_base or with T=0.05 (branch exp/rare-class-sampling 15bfb29).
- Decision (user, 2026-10-01): exception to the keep bar. At the 5-win off-road check, run exp_base and exp_base + cherry-pick 15bfb29 on the off-road preset; keep it if the off-road ΔmIoU ≥ 0.005.
- Off-road A/B/C (2026-10-02, GOOSE 30 ep): A baseline 640² 0.625 / 0.623 (0.624); B exp_base 3c37232 0.644 / 0.632 (0.638, +0.014 vs A); C = B + RCS (d0e7846) 0.650 / 0.643 (0.6465, torch 0.6498 / 0.6433) → +0.0085 vs B @ 1.88 / 1.87 ms, both seeds up. ✅ ff-merged d0e7846 into exp_base. GOOSE rare classes are far rarer (animal 2→23 % of draws, human 12→42 %, water 6→25 %). exp_base+RCS has no Cityscapes row yet: re-baseline before batch 2.

## boundary-label-relaxation — ❌
- Change: in `SemSegCriterion`, for pixels whose 3×3 GT neighbourhood contains > 1 class (max-pool the
  one-hot with ignore masked), replace CE with `−log Σ_{c ∈ N(p)} softmax_c`; interior pixels keep CE.
  Apply to `loss_ce` and `loss_aux`. Knob `train.sem_seg.boundary_relax: 1` (radius px, 0 = off).
  `dfine_seg/model/sem_seg_criterion.py`, test in `tests/unit/test_sem_seg.py`.
- Why: our label map is the 2048×1024 PNG NEAREST-downsampled 3.2×/1.6× (and warped again in mosaic), so
  boundary pixels are aliased noise; CE forces the network to fit that noise and Dice then argues with
  it. Relaxation only demands that the boundary pixel belong to one of its adjacent classes. Zhu et al.
  (CVPR 2019, arXiv:1812.01593) report +1.4 mIoU on Cityscapes with relaxation in their full recipe.
  If it fails, the alternative boundary lever is STDC's train-only detail head (binary Laplacian boundary
  aux on the stride-8 PAN feature).
- Latency risk: none.
- Result: r=1 (4.6 % of valid pixels relaxed), +8 % train step. Seed 42 TRT 0.732 (torch 0.7331) vs base s42 0.736 / 0.7353 → −0.004 @ 1.86 ms; seed 123 skipped (would need 0.756, +0.014 over base s123, to reach the 0.744 bar). ❌ no gain; the downsampled-boundary noise isn't what limits mIoU here (branch exp/boundary-label-relaxation edce3c8). STDC detail-head alternative not tried.

## ema-horizon — ❌
- Change: give the EMA a real averaging window. `ModelEMA` uses `0.9998·(1 − e^{−it/2000})`; on the
  7 440-iteration screen the momentum ends at 0.9756 → a 41-iteration (0.2-epoch) horizon, i.e. no
  averaging (on GOOSE, 14 700 it, it reaches 2.4 epochs). New `train.ema_horizon_epochs: null` (legacy;
  task default 2 for sem_seg): `m = 1 − 1/(h·steps_per_epoch)`, warmup `tau = 1/(1 − m)`. Files:
  `dfine_seg/dl/train.py` (`ModelEMA`, Trainer init), `config.yaml`, `default.yaml`.
- Why: weight averaging over the annealing tail is a free +0.2…0.5 mIoU on segmentation (Polyak
  averaging; Tarvainen & Valpola 2017; timm `ModelEmaV2` defaults 0.9998 for ~100k-iteration runs,
  i.e. ~5 % of training, vs our 0.5 %). The July effb5 diagnosis flagged the near-identity EMA but the
  knob was reverted untested; the detection "#6 EMA bracket" was planned and never run. Flag: shared
  `ModelEMA`; null default keeps detection identical.
- Latency risk: none.
- Result: horizon 2 ep (m 0.99731, tau 372; final window 372 it vs 41). TRT 0.738 / 0.734 (mean 0.736, torch 0.7379 / 0.7342) vs 0.739 → ΔmIoU −0.003 @ 1.87 / 1.88 ms. ❌ longer averaging doesn't help; at 40 ep the annealed LR tail already does the smoothing (branch exp/ema-horizon 0015b37).

## stride4-detail-skip — ❌ reverted 2026-10-03 (the +0.005 was run-to-run noise, see Follow-ups)
- Change: feed HGNetv2 stage1 (stride 4, 64 ch, already computed) into the seg head. `build_model`
  (`dfine_seg/model/dfine.py`): for sem_seg add stage index 0 to `return_idx` (S/M/L/X; nano keeps its
  1/8 path); `DFINE.forward` already hands the extra leading feature over as `low_level_feat`.
  `SemSegDecoder`: if the low-level feature is at 1/4, do **not** prepend it to the fuser (that moves
  `fusion_conv` to 1/4 = +15 GMAC); instead `GN(1×1 conv 64→256)` and add it after MaskDecoder's
  bilinear ×2 and before `up_conv`, via a new `skip=None` arg on `MaskDecoder.forward`
  (`dfine_seg/model/arch/dfine_decoder.py`). New params are missing from the COCO checkpoint like the
  neck already is. Test `test_decoder_shapes_and_aux`.
- Why: the head predicts at 1/4 but its finest input is stride 8; the ×2 upsample invents detail. A
  stride-4 lateral is the U-Net/FPN-P2/BiSeNetV2-detail-branch fix, and the thin classes it helps
  (pole, sign, light, fence, rider) are exactly the S deficit. The nano 1/8→1/4 change gave +11 %
  relative mIoU (CHANGELOG 2026-02-28). Different from the shelved July "Track B": placed after the
  fuser (no 1/4 fusion cost) and motivated by Cityscapes thin classes, not Semantic-Drone. Flag: touches
  `MaskDecoder` (shared with instance seg) with a default-None arg; run the seg regression only if asked.
- Latency risk: low. +0.42 GMAC (1×1 64→256 at 160²) + one add, vs ≈ 30 GMAC head → ≈ +1–2 % engine.
- Result: GN-free 1×1 conv (64→256, zero-init weight) added after MaskDecoder's ×2; GN variant was 1.94 ms (> limit). TRT 0.742 / 0.746 (mean 0.744, torch 0.7419 / 0.7464 → +0.0051) vs 0.739 → ΔmIoU +0.005 @ 1.89 ms (1.013×). ✅ borderline (exactly the bar, both seeds up); COCO-init bias runs against it (new layer cold). Checkpoints with/without skip told apart by `decoder.detail_proj.weight`. ff-merged 3c37232.

## offset-learning-head — ❌
- Change: OffSeg offset learning on top of the existing classifier, sem_seg only
  (`SemSegDecoder` in `dfine_seg/model/arch/dfine_decoder.py`). With neck output E (N×128 at 1/4) and
  classifier weights W (19×128): A = W·Eᵀ; class offsets ΔW = MLP(softmax_N(A)·E), feature offsets
  ΔE = MLP(softmax_K(A)ᵀ·W); logits = (W+ΔW)·(E+ΔE)ᵀ. Matmul/softmax/MLP only, no grid_sample.
- Why: per-pixel classifiers use one fixed prototype per class; OffSeg adapts prototypes per image and
  nudges pixel features toward them. Plug-in gains of +2.7 / +1.9 / +2.6 mIoU on SegFormer-B0 /
  SegNeXt-T / Mask2Former-T with 0.1–0.2 M params (Zhang et al., ICCV 2025, arXiv:2508.08811,
  github.com/HVision-NKU/OffSeg). Unverified on a conv head like ours, hence ranked below the cheap
  losses.
- Latency risk: low–medium. ≈ 1 GMAC (per-pixel MLP 128→128→128 at 160² dominates; the two K=19 matmuls
  are 0.06 GMAC each) → ≈ +2–3 % engine. Softmax over 25.6 k positions in fp16: export already keeps
  Softmax fp32 on TRT 11, so fine. Fallback if > 1.03×: compute the offsets on a 2× avg-pooled E.
- Result: reference-code formulation (bias-free Linear P, Q; zero-init; no output LayerNorm; main head only), factored to ≈0.19 GMAC. Seed 42 TRT 0.729 (torch 0.7292) vs base s42 0.742 / 0.7419 → −0.013 @ 1.94 ms (at the limit); seed 123 skipped. ❌ clear loss on a conv head + 40-ep screen (branch exp/offset-learning-head 3afa1ab).

## photometric-augs — ❌
- Change: real colour augmentation for sem_seg: `A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.5)` (or
  `A.HueSaturationValue`) in `SemSegDataset._init_augs` (`dfine_seg/dl/dataset.py`) behind a new
  `train.augs.color_jitter: 0.5` knob; current photometric knobs stay as they are.
- Why: today's photometric augs are essentially off (brightness p 0.02, gamma 0.02, blur 0.01, noise
  0.01, gray 0.01); the only real regulariser is mosaic. PhotoMetricDistortion is in every mmseg
  Cityscapes recipe and in DDRNet/PIDNet/SegFormer training. Off-road lighting (seasons, dusk, glare) is
  the transfer axis we care about. Different from the July "effb5-augs" loss (−0.023 on Semantic-Drone):
  that was heavy geometric + photometric; this is mild, photometric only. Flag: the knob lives in the
  shared `augs:` block; `CustomDataset` ignores it unless wired (say so in the config comment).
- Latency risk: none.
- Result: ColorJitter(0.4, 0.4, 0.4, 0.1) p=0.5, also on mosaic samples. TRT 0.737 / 0.736 (mean 0.7365, torch 0.735 / 0.7368) vs 0.744 → ΔmIoU −0.0075 @ 1.89 / 1.90 ms. ❌ on Cityscapes (photometrically homogeneous, so jitter is pure regularisation cost at 40 ep). Untested where it could matter: off-road lighting variety — only worth an off-road-only A/B if we ever screen there (branch exp/photometric-augs 8f31bec).

## backbone-lr-respect — ❌
- Change: `respect_backbone_lr = cfg.model_name in ("l", "x") or cfg.task == "sem_seg"` in
  `Trainer.__init__` (`dfine_seg/dl/train.py`). One line; uses the existing l/x code path (backbone
  groups on AdamW at `backbone_lr·2` = 1.2e-4 peak instead of the Adan aux peak).
- Why: for S the OneCycle peak list is `[aux_peak]*4`, so the COCO-pretrained backbone runs at
  2.5e-3 (base_lr·2·adan_lr_mult), 40× its nominal 6e-5, for a 40-epoch dense fine-tune. Detection
  "#10 backbone-LR" was a tie, but that was ImageNet init, a walltime cap and box supervision; here the
  backbone is the pretrained part we want to keep. Two things move together (optimizer + LR) because
  that is the shipped l/x path. Low prior, trivial cost. Flag: shared Trainer code, task-guarded.
- Latency risk: none.
- Result: verified before: backbone Adan peak 2.5e-3 → AdamW 1.2e-4 (21× lower); all 8.01 M params grouped once. Seed 42 TRT 0.732 (torch 0.7314) vs base s42 0.742 / 0.7419 → −0.010 @ 1.88 ms; seed 123 skipped. ❌ the high shared Adan LR is right for this dense fine-tune; a lower backbone LR underfits at 40 ep (branch exp/backbone-lr-respect e97e4f7).

## cwd-distill-from-m — ✅ parked (stack at end)
- Change: channel-wise distillation from a frozen D-FINE-seg M teacher (same 640 squish preprocessing,
  e.g. `cityscapes/output/models/sem_seg_m_1280_simple_2026-07-18` = 0.783 TRT, keep_ratio false):
  KL between channel-wise (per-class, over pixels) softmax of teacher and student 1/4 logits, T = 4,
  weight 3, added to `SemSegCriterion`. Knob `train.sem_seg.kd_teacher: <path>` (null = off); teacher
  forward under `no_grad` in `Trainer.train` for sem_seg only. Files: `dfine_seg/dl/train.py`,
  `dfine_seg/model/sem_seg_criterion.py`, `config.yaml`, `default.yaml`.
- Why: CWD is the strongest published train-only lever for real-time seg (PSPNet-R18 70.1→74.9 on
  Cityscapes, Shu et al., ICCV 2021, arXiv:2011.13256); student–teacher headroom here is 0.73 vs 0.79,
  unlike the VisDrone detection KD (teacher had no headroom). Teacher is our own model, so it is allowed
  under the fair-comparison rule. Ranked last: ~1.5× train time per seed, and the off-road check needs
  its own M teacher trained first (~2 h) — an operational cost every new dataset pays.
- Latency risk: none (train-only).
- Result: teacher sem_seg_m_896_2026-07-18 (square, TRT 0.783; the 640×1280 M ckpts are pre-pos-embed-fix). CWD on 1/4 logits, T 4, w 3. TRT 0.752 / 0.747 (mean 0.7495, torch 0.7519 / 0.7475) vs 0.744 → ΔmIoU +0.0055 @ 1.88 / 1.89 ms (deploy graph unchanged). Cost: train 67 vs 52 min/seed (1.3×), VRAM 97 %, needs a trained M teacher per dataset (off-road preset fails without one). Branch exp/cwd-distill-from-m c075240.
- Decision (user, 2026-10-02): not merged into exp_base, so screens stay cheap and teacher-free. At the end, stack on the final exp_base and verify on Cityscapes + off-road (needs a GOOSE M teacher, square input).

# Follow-ups — 2026-10-02 → 10-04 (branch `review`, see followups.md)

All TRT fp16 mIoU on the presets above; rows in results.tsv, runs in experiments/runs/<preset>/<name>/.
- **Run-to-run noise (decisive finding).** repro-exp_base = exact 3c37232, same seeds: 0.732 / 0.741 vs the
  original 0.742 / 0.746. rebase-norcs (review code, RCS off; code path == 3c37232 bar renames): 0.737 / 0.744.
  Same config + seed moves up to 0.010; per-run sd ≈ 0.005 on Cityscapes (GOOSE ≈ 0.0015). Protocol changed to
  3 seeds and a 0.006 bar (program.md). Batch-1 wins below ~0.008 were not real.
- **stride4-detail-skip, re-measured → removed.** Cityscapes: with 0.740 (6 runs) vs without 0.739 (2 runs).
  GOOSE (review, rot 0, no sampling): goose-str4 0.632 / 0.633 / 0.635 (0.6333) vs goose-nostr4 0.638 / 0.632
  (0.635). Instance seg (MaskDecoder version, regression seg config, 180-min cap = 21 ep, vs regr_seg_2026-10-01
  trajectory): f1 0.647 vs 0.648, iou 0.368 vs 0.368, mask mAP50 0.445 vs 0.471 (−0.026). Removed for both tasks.
- **Rare-class sampling.** Cityscapes 2×2 (stride-4 on, 2 seeds): rot 0.05 — no RCS 0.740 (6 runs) / RCS 0.7355;
  rot 0 — no RCS 0.737 / RCS 0.7425 → no consistent effect. RCS on GOOSE never draws 14 % of images (eff. unique
  24 %, vegetation pixel share 58 → 45 %), too aggressive (user: deployment sees the natural distribution).
  LVIS repeat-factor sampling (RFS, t = 1, r_c = 1/√(image share)):
  - detect, Cityscapes regression config 55 ep, rfs_det_{false,true}_s{42,123}: best f1 0.6894 vs 0.6874,
    mAP50 0.5978 vs 0.6004, iou 0.4318 vs 0.4296, TRT f1 0.693 / 0.694 vs 0.689 / 0.692; rare-class f1 worse
    (train −0.074, bus −0.039, truck −0.027). ❌
  - sem_seg GOOSE (stride-4 on, rot 0): goose-str4-rfs 0.638 / 0.638 / 0.636 (0.6373, px 0.9477) vs 0.6333
    (px 0.9480) → +0.004, won every seed pairing, below the 0.006 bar.
  - sem_seg Cityscapes, final code (no stride-4, rot 0): final-rfs 0.733 / 0.735 / 0.729 (0.7323) vs final-base
    0.733 / 0.732 / 0.733 (0.7327), px 0.9503 both → flat.
  Decision (user, 2026-10-04): keep RFS for all tasks, `train.rare_class_sampling: false` by default; RCS deleted.
- **Pos-embed / anchor per-step rebuild cache:** train step −2.4 % detect, 0 sem_seg / segment → reverted.
- Current Cityscapes reference for batch 2 (final code, rot 0, no sampling): final-base 0.7327 (3 seeds) @ 1.87 ms.

# Batch 2 — 2026-10-02

Baseline `exp_base` 3c37232 (native aspect 448×896 + stride-4 skip): TRT mIoU 0.742 / 0.746 (mean 0.744)
@ 1.89 ms. Keep bar: ΔmIoU ≥ +0.005 at ≤ 1.94 ms (1.03×); a speed win is ≤ 1.80 ms (−5 %) at ΔmIoU ≥ −0.001.
Batch 1 taught: input geometry (+0.0085) and real stride-4 detail (+0.005) win; loss reweighting (OHEM −0.006,
boundary relaxation −0.004), added regularisation (photometric −0.0075, long EMA −0.003), a lower backbone LR
(−0.010) and a prototype head (OffSeg −0.013) all lose on the 40-epoch COCO-init screen; seed noise ≈ 0.004–0.006.
Parked for the end (don't re-propose): rare-class sampling, CWD from M. Ranked by expected gain ÷ cost.

Measured for this batch (CPU MAC count of the deploy graph at 448×896; per-class IoU from
`experiments/runs/cityscapes/stride4-detail-skip/*/extended_metrics.csv`):
- 40.9 GMAC total, of which the seg head is 30.6 (up_conv 14.8, neck 11.1, fusion_conv 3.7, detail_proj 0.4),
  encoder 7.8, backbone 2.5. The head's 3×3 convs run near fp16 peak (≈ 0.012 ms/GMAC); everything else is
  launch/memory-bound, so an extra *kernel* costs more than an extra GMAC. One GroupNorm(32, 256) at 1/4 measured
  +0.05 ms (stride4-detail-skip entry); the sem_seg head has seven GNs.
- Weakest classes (mean of seeds): wall 0.54, pole 0.55, fence 0.55, motorcycle 0.55, rider 0.61, terrain 0.64,
  traffic-light 0.64 — thin structures and rare classes. Rare-class seed spread is 0.03–0.07 (motorcycle
  0.517/0.585, bus 0.824/0.871), so single-seed rare-class deltas are noise.
- Muon currently drives only the one AIFI layer for sem_seg (`_is_muon_param`: ndim == 2 and no "mask" in the
  name, `dfine_seg/model/dfine.py`); every head conv is on Adan at the shared 2.5e-3 peak.
- `train.augs.rotate_90: 0.05` with `A.Affine(fit_output=True)` *before* `A.Resize` turns a 1024×2048 frame
  into 2049×1027 portrait (verified, albumentations 2.0.8), then squishes it 4.6:1 — 5 % of training samples.

## drop-rotate90 — pending
- Change: `train.augs.rotate_90: 0.05 → 0.0` in `config.yaml` and `dfine_seg/config/default.yaml` (keep
  `tests/unit/test_config_template.py` happy). One number, no code.
- Why: `SemSegDataset._init_augs` (`dfine_seg/dl/dataset.py`) rotates 5 % of samples — plain and mosaic alike —
  by 90° with `fit_output=True` and only then resizes to 448×896, so those frames are driving scenes turned
  sideways and distorted to an aspect never seen at test time; their thin-class labels are near-noise after the
  4.6× vertical squish. Cameras in Cityscapes and GOOSE have a fixed orientation, so this is a dataset-family fact,
  not a Cityscapes trick; the knob came from the aerial Semantic-Drone work where 90° rotations are valid.
  `A.Rotate(limit=10, p=0.05)` is harmless and stays. Expected +0.001–0.004 (5 % clean samples back, no capacity
  spent on an impossible orientation). Flag: shared `augs:` block — detection presets inherit the new default;
  aerial presets can set 0.05 locally.
- Latency risk: none.
- Result (2026-10-02, review e5bf0fa = native + stride-4 + RCS, Cityscapes): TRT 0.744 / 0.741 (mean 0.7425) vs
  rebase-rcs 0.733 / 0.738 (0.7355) → +0.007 @ 1.89 ms, pixel_acc 0.952 both. Without RCS the sign flips
  (norcs-norot 0.737 / 0.737 vs rebase-norcs 0.737 / 0.744), so the effect is within noise (see Follow-ups).
  ✅ adopted as the default anyway (user, 2026-10-02): 90° rotation is wrong for fixed-orientation cameras;
  aerial presets can set it locally.

## blv-logit-variation — pending
- Change: Balancing Logit Variation inside `SemSegCriterion` (`dfine_seg/model/sem_seg_criterion.py`), train-only:
  per pixel and class `z_k += (c_k / max_i c_i) · clamp(|N(0, σ²)|, 0, 1)` with `c_k = log(Σ_j q_j / q_k)`, `q_k`
  = pixel count of class k (running sum of the batch histograms in a `register_buffer`; the paper counts the train
  split once, the running sum converges within an epoch), σ = 4 (their ablation: 4–6 optimal, flat within 1 %).
  Apply to `loss_ce` and `loss_aux`; Dice and eval/export see raw logits. Knob `train.sem_seg.blv_sigma: 4.0`
  (null = off) in `config.yaml` + `default.yaml`, plumbed through `build_loss` (`dfine_seg/model/dfine.py`). Test
  in `tests/unit/test_sem_seg.py` (off → identical loss; on → finite, eval unchanged).
- Why: Wang et al., "Balancing Logit Variation for Long-tailed Semantic Segmentation", CVPR 2023
  (arXiv:2306.02061), Table 1: Cityscapes val, seven architectures (HRNet-18/OCR +0.72, R50/UPerHead +0.35,
  R50/PSP +0.55, R101 +0.47, MiT-b0 +0.24, Swin-T +0.43, ViT-B +1.20), tail classes (wall, light, sign, rider,
  truck, bus, train, m.bike, bike) +1.2 … +3.2, no parameters, discarded at inference. Our weakest non-thin classes
  are exactly that tail set (wall 0.54, motorcycle 0.55, rider 0.61, truck/train 0.76). Not loss reweighting (OHEM,
  lost) and not sampling (RCS, parked): a class-scaled stochastic margin on the logits. Their Table 5 puts plain
  Logit Adjustment *below* the baseline on Cityscapes (75.9 vs 76.5) and Lovász at +0.1, so don't try LA / Balanced
  Softmax / inverse-frequency `class_weights` here. Expected +0.002–0.005 (their gains are on full-res crops with
  40–160 k iterations; ours is a 40-epoch low-res screen with Dice already present).
- Latency risk: none.
- Result: pending

## head-bn-fold — pending
- Change: build the sem_seg head with BatchNorm instead of GroupNorm so every norm folds into its conv at export.
  `MaskDecoder(in_chs, out_ch, norm="gn")` gets a `norm` arg (`"bn"` → `nn.BatchNorm2d` for the three laterals,
  `fusion_norm`, `bn1`; default GN keeps instance seg byte-identical); `SemSegDecoder` passes `norm="bn"` and
  builds `neck` / `aux_head` from a `conv_bn_act` (`dfine_seg/model/arch/dfine_decoder.py`). `describe()`
  (`dfine_seg/api/ckpt.py`) reports `head_norm` from the presence of `decoder.mask_decoder.bn1.running_mean`
  (weights, not meta) and `build_model` / `dfine_seg/dl/export.py` / `dfine_seg/infer/torch_model.py` /
  `torch_compile_model.py` pass it like `detail_skip`. COCO GN affine (`weight`/`bias`, shape [C]) loads into
  the BN affine unchanged; running stats start at 0/1 and are re-estimated from the first batch (EMA already
  averages BN buffers through `state_dict()`). Tests: shapes, a GN checkpoint still builds, BN checkpoint
  round-trips strict. Executor pre-check: export + bench a randomly initialised BN build before training — TRT
  latency is weight-independent, so the speed verdict is known in 10 minutes.
- Why: a latency win, not an accuracy one. GN is a two-pass kernel TRT cannot fuse (one GN(32, 256) at 1/4 =
  +0.05 ms measured); the head has three GNs at 1/4 (bn1, neck ×2) and four at 1/8 (laterals, fusion_norm). BN in
  eval is an affine map that TRT folds into Conv+BN+ReLU, so all seven kernels disappear: estimate −0.12 … −0.18 ms
  (1.71–1.77 ms, −6 … −9 %), which clears the program's "≥ 5 % faster at ΔmIoU ≥ −0.001" win on its own and buys
  the headroom the accuracy ideas below need (the GN'd stride-4 projection that was over budget, #subpixel,
  #stride2). Accuracy: BN is the head default in mmseg, PP-LiteSeg, DDRNet, PIDNet and STDC; with batch 16 ×
  25 k positions per sample the per-channel statistics are far better estimated than in detection, and the fuser
  used GN for instance seg's small effective batch, not for dense labels. Expected ΔmIoU within ±0.003. Flag:
  `MaskDecoder` is shared with instance seg (default-preserving arg); arch change, so COCO-init bias applies
  (the fuser's normalisation statistics change at step 0).
- Latency risk: none (negative).
- Result: pending

## stdc-detail-head — pending
- Change: train-only STDC Detail Guidance on the stride-4 backbone feature the skip consumes. `SemSegDecoder`
  gets `detail_head = Conv3×3(C_str4 → 64)-BN-ReLU + Conv1×1(64 → 1)` on `low_level_feat` (stage 1, 64 ch at
  1/4), output bilinear ×4 to input res, emitted as `outputs["detail_logits"]` only when `self.training`.
  Detail GT built on the fly in `SemSegCriterion` from `sem_mask` (448×896): for s ∈ {1, 2, 4} take
  `label[:, ::s, ::s]`, mark a pixel as detail if its 3×3 window contains two classes (max-pool/min-pool of the
  one-hot, windows touching ignore=255 are masked out), nearest-upsample back by s, OR the three scales — the
  parameter-free form of STDC's Laplacian-pyramid + learnable 1×1 + 0.1 threshold. `loss_detail = BCEWithLogits +
  Dice` (ε = 1, batch-level, masked), weight 1.0 → add `"loss_detail": 1` to `SemSegCriterion.weight_dict` in
  `dfine_seg/model/configs.py`. Files: `dfine_seg/model/arch/dfine_decoder.py`, `dfine_seg/model/sem_seg_criterion.py`,
  `dfine_seg/model/configs.py`, `tests/unit/test_sem_seg.py`. Deploy graph unchanged; construct the head
  unconditionally like `aux_head` (used only when `self.training`), so checkpoints carry its keys and the strict
  loads in infer/export keep working without a `describe()` change.
- Why: Fan et al., "Rethinking BiSeNet" (STDC, CVPR 2021, arXiv:2104.13188) Table 4, Cityscapes val at 512×1024
  input — the same low-resolution regime as our 448×896: STDC2-50 73.0 → 73.8 with the 1× detail GT, 74.2 with
  1×+2×+4× (+1.2), identical FPS; their Fig. 5/6 show the guided low-level feature encodes boundaries and corners
  and fixes small objects. PIDNet (CVPR 2023) relies on the same boundary supervision of its detail branch.
  Different from boundary-label-relaxation (which *removed* supervision at boundaries and lost): this *adds* a
  dense binary boundary task on the exact feature the stride-4 skip feeds. Our lateral is a 1×1 on raw stage-1
  features pretrained for detection, which never needed sharp edges; a boundary task makes stage 1's three convs
  keep them. Target classes: pole / fence / wall / sign (0.54–0.71). Expected +0.003–0.008.
- Latency risk: none (train-only).
- Result: pending

## subpixel-logit-residual — pending
- Change: predict logits at 1/2 instead of 1/4 for 0.24 GMAC. In `SemSegDecoder`: `self.subpix =
  nn.Conv2d(neck_dim, 4·num_classes, 1)` zero-init (weight and bias); `l4 = classifier(x)`; `l2 =
  F.interpolate(l4, scale_factor=2, bilinear) + F.pixel_shuffle(self.subpix(x), 2)`; `logits =
  F.interpolate(l2, scale_factor=2, bilinear)` (full res, as today). Zero-init → starts as the exact current model.
  `decoder.classifier.weight` keeps `num_classes` for `describe()`; new key `decoder.subpix.weight` → a `subpix`
  flag in `describe()` passed through `build_model` / export / infer wrappers like `detail_skip`, so older
  checkpoints still build. PixelShuffle exports as ONNX DepthToSpace (TS exporter), native in TRT. Files:
  `dfine_seg/model/arch/dfine_decoder.py`, `dfine_seg/model/dfine.py`, `dfine_seg/api/ckpt.py`,
  `dfine_seg/dl/export.py`, `dfine_seg/infer/torch_model.py`, `torch_compile_model.py`, `tests/unit/test_sem_seg.py`.
- Why: the output grid is 112×224 — one cell is 9.1×9.1 px of the 2048×1024 frame, wider than most poles (8–15
  px) and sign posts; bilinear ×4 can place a boundary between cells but cannot draw a structure narrower than a
  cell. A sub-pixel classifier (DUpsampling, Tian et al., CVPR 2019, arXiv:1903.02120: label patches are linearly
  reconstructible from a per-cell code; sub-pixel conv is learned DUpsampling) lets each cell emit four
  position-specific logit vectors from the same 128-d feature, which — via the stride-4 lateral — now carries
  sub-cell edge evidence. The July upsample A/B (+0.004, all in pole/sign/light) showed finer output placement pays;
  this is the in-budget route (the rejected variant resized 19 channels to 2048×1024, +20 %). Expected
  +0.002–0.006, concentrated in pole/fence/light/sign. Risk: 2×2 checkerboard (the ×2 bilinear smooths it, CE at
  full res trains the four sub-classifiers jointly).
- Latency risk: low — 1×1 128→76 at 1/4 (0.24 GMAC ≈ 0.003 ms), DepthToSpace + Add on 1.9 M elements, and the
  final Resize writes the same 7.6 M outputs as today: ≈ +0.02–0.03 ms (≤ 1.5 %). Verify; if > 1.03× run it on
  top of #head-bn-fold.
- Result: pending

## soft-label-downsampling — pending
- Change: supervise with class-fraction labels instead of point-sampled ones (train mode only). `SemSegDataset`
  (`dfine_seg/dl/dataset.py`) returns the label at 2× (896×1792): plain path — run `A.Compose(augs)` at native
  res, then `cv2.resize(image, (W, H), LINEAR)` (unchanged deploy kernel) + `cv2.resize(mask, (2W, 2H), NEAREST)` +
  normalize/ToTensor; mosaic path — build `mask4` on a 4H×4W canvas (tiles resized to (2W, 2H)) and warp it with
  `M2 = S·M·S⁻¹`, `S = diag(2, 2, 1)`, `dsize = (2W, 2H)`. `SemSegCriterion`: `one_hot(mask2x)` (ignore → all
  zero) → `avg_pool2d(2)` → soft target `q` at H×W, `valid = q.sum(1)` (fraction of labelled sub-pixels);
  `loss_ce = −Σ_c q_c log p_c` per pixel, weighted by `valid`; Dice with `q` in place of the one-hot; aux CE the
  same. Val/bench untouched (1× hard label, full-res eval). Knob `train.sem_seg.soft_labels: true` (false restores
  today's path) in `config.yaml` + `default.yaml`. Also `_debug_image` (downsample the 2× mask for the overlay) and
  tests `test_load_mosaic_target_size_and_ignore_fill`, `test_augs_preserve_class_ids`, criterion tests.
- Why: our label is the 2048×1024 PNG NEAREST-downsampled 2.29× (then NEAREST-warped again in mosaic): a pole
  edge survives as a random 0–2 px, a distant sign vanishes or doubles, while the eval scores the full-res hard
  label — the net trains on a thinner, noisier version of the test target. Soft labels keep the class proportions
  inside each cell: "Soft labelling for semantic segmentation: bringing coherence to label down-sampling"
  (arXiv:2302.13961, 2023/24; Cityscapes at 1/4 and 1/8 res, DeepLabV3+/HRNet, gains concentrate on fence, pole,
  wall — their headline +11 mIoU at 256×512 is under their own soft/low-res scoring, not our full-res hard-label
  protocol, so expect a fraction), and DSRL (Wang et al., CVPR 2020) gets ≥ +2 mIoU by supervising a 0.5×-input
  model against full-res labels. The 2× hard label + 2×2 pool gives 5-level fractions (0, ¼, ½, ¾, 1), enough to
  say "thin thing present", at zero deploy cost. Targets pole / fence / light / sign. Expected +0.003–0.008; ranked
  below #stdc because it is ~80 lines across dataset + criterion. Memory: 2× label 12.8 MB/sample int64
  (205 MB per batch) plus a transient ~1 GB one-hot; no logits upsampling beyond today's ×4.
- Latency risk: none (train-only).
- Result: pending

## head-dropout-off — pending
- Change: `SemSegDecoder(dropout=0.1 → 0.0)` — the `Dropout2d` before the classifier and inside `aux_head`
  (`dfine_seg/model/arch/dfine_decoder.py`; sem_seg-only code). One default.
- Why: `Dropout2d(0.1)` zeroes 10 % of the 128 neck channels per sample — whole feature maps — so the classifier
  trains on a randomly thinned basis while the EMA/eval model sees all channels; it is the only stochastic op
  between the fuser and the logits and makes the CE/Dice gradient noisier for rare classes carried by few channels.
  It is mmseg's default for 80–160 k-iteration schedules and was inherited untested. Every added regulariser lost
  on this 40-epoch COCO-init screen (photometric −0.0075, boundary relaxation −0.004, long EMA −0.003) while the
  under-regularised, higher-LR side won (backbone-lr-respect −0.010 the other way), which says the screen sits on
  the under-fitting side of the regularisation optimum. Expected +0.000–0.004; if it loses, dropout 0.2 is the
  one-number follow-up and settles which side we are on.
- Latency risk: none (identity at eval).
- Result: pending

## muon-head-convs — pending
- Change: route the sem_seg head's conv kernels to Muon. `dfine_seg/model/muon.py`: `_muon_update` flattens
  `grad.view(grad.size(0), -1)` (and the momentum buffer) for ndim > 2 — `_zeropower_via_newtonschulz5` asserts
  2-D today and `upd.reshape(p.shape)` is already there. `_is_muon_param` (`dfine_seg/model/dfine.py`): for sem_seg
  also accept 4-D weights named `decoder.mask_decoder.lateral.*`, `decoder.mask_decoder.fusion_conv`,
  `decoder.mask_decoder.up_conv`, `decoder.neck.*` (Keller Jordan's rule: hidden layers only — `classifier`
  (output), `detail_proj` (zero-init), `aux_head`, norms and biases stay on Adan). Same Muon group → peak LR 5e-3
  (`muon_lr·2`), momentum 0.95, wd 1.25e-4. `build_optimizer` gets `task=` from `Trainer` (or detects
  `decoder.neck`). Knob `train.muon_head_convs: true`, default false → detection/instance byte-identical.
- Why: the detection campaign's only movers were optimizer changes (Muon +0.004, Adan +0.005 mAP, both confirmed
  at X/75 ep), yet for sem_seg Muon touches only the AIFI layer (0.5 M of 8 M params) while the 19 GMAC of
  up_conv/fusion/neck sit on Adan. Muon on conv filters flattened to (out, in·k·k) set the CIFAR-10 speedrun record
  (kellerjordan.github.io/posts/muon, `airbench94_muon.py`) and is the Moonlight/Kimi practice for every hidden
  matrix. Different from the rejected `muon-wd` / `moonlight-rms` (update shaping of the existing group) and
  `backbone-lr-respect` (LR ratio): this changes *which* parameters get the orthogonalised update. Caveat: Muon's
  RMS-matched step on a (256, 2304) kernel is ≈ 0.02·lr per element — 5–20× smaller than Adan's at 2.5e-3 — so if
  seed 42's train loss trails the baseline at epoch 10, rerun with the conv group at `muon_lr·4` before rejecting.
  Expected −0.005 … +0.006. Flag: shared `muon.py` and `_is_muon_param` (task-guarded).
- Latency risk: none.
- Result: pending

## stride2-logit-refine — pending
- Change: an edge-conditioned logit correction at 1/2 from the only feature with ≥ 2 px resolution at 448 scale.
  `StemBlock.forward` (`dfine_seg/model/arch/hgnetv2.py`) stashes the `stem1` output (16 ch, stride 2) when a
  `return_stem2` flag is on; `HGNetv2.forward` returns it first; `DFINE.forward` (`dfine_seg/model/dfine.py`)
  passes it as `stem_feat`; `SemSegDecoder`: `refine = nn.Conv2d(16, num_classes, 1)` zero-init, `l2 =
  up2(l4) + refine(stem_feat)`, `logits = up2(l2)`. New key `decoder.refine.weight` → `describe()` flag like
  `detail_skip`. Files as in #subpixel plus `hgnetv2.py`.
- Why: Gated-SCNN / DeepLabV3+-style low-level guidance, one stride finer than anything the head sees: `stem1` is
  learned edge/colour filters on the raw image, so a zero-init 1×1 adds an edge-dependent logit offset at 1/2 — a
  learned guided-filter step for boundaries that #subpixel (which only redistributes the 1/4 evidence) cannot
  provide. Run only if #subpixel wins (then this is the next detail step); if #subpixel loses, skip this too.
  Expected +0.002–0.005 on pole/fence/light/sign.
- Latency risk: low–medium — 1×1 16→19 at 224×448 (0.03 GMAC) plus an extra Resize + Add at 1/2: ≈ +0.03–0.05 ms
  (1.5–2.5 %); pair with #head-bn-fold if it lands near the limit. Flag: touches the shared HGNetv2 stem
  (default-off flag).
- Result: pending

Considered, not proposed: plain Logit Adjustment / Balanced Softmax / inverse-frequency `class_weights` (BLV Table 5:
LA below baseline on Cityscapes); Lovász (−0.009 on Semantic-Drone, July); learned non-uniform downsampling (ES,
Marin et al., ICCV 2019 — the strongest low-res lever in the literature, but it needs a warp in every infer wrapper
and the C++ path); HGNetv2-B1 backbone swap (same channel plan as B0, +2.5 GMAC, but no COCO weights → COCO-init
bias); any engine-side or logits-to-original-res change (rejected in July, see HISTORY).
