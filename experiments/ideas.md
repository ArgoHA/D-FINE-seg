# Idea backlog — candidate improvements (researched 2026-06-13)

**Run queue for the autoresearch loop.** Read this + `lab_notebook.md`, then run one experiment at a time
following `EXPERIMENT_GUIDE.md` §5 (branch → single change → `make test` → `run_candidate.py` → `promote.py`
→ notebook). One change per experiment; read the paper first. Approval gate (guide 1.8) applies.

**Mission (guide §0):** improve D-FINE-seg generally — VisDrone is only the screen. Every idea must improve
the recipe *as users run it* (full schedules). Screen-regime-only tweaks are methodology, not candidates.

**Baseline to beat (horizon-30):** test mAP_50_95 **0.2167**, f1 **0.5635** (Adan, promoted 2026-06-14).
Margins = floor **0.003**. (Pre-Adan Muon baseline was 0.2119/0.5565; horizon-100 was 0.2061/0.552.)

## Diagnosis

- Run shape: ~809 steps/epoch (6,471 imgs, bs 8) × ~21-22 epochs under the 60-min cap ≈ **18k steps**.
  Warmup 3 epochs, ends ~65-80% through the anneal. Two constants: **mosaic never closes** (user decision);
  **EMA τ = 5k steps ≈ 28% of run**.
- **Per-step gradient quality is the proven lever** (Muon + Adan won). Supervision density falsified 3×.
  Matcher-cost, optimizer-shaping, and cls-target families all probed and exhausted at Tier 1.
- Dataset: 343,205 boxes; mean 53 / median 42 / p95 132 / max 902 objects/image; **55.4% < 16 px** (21.9% < 8 px);
  class imbalance ≈ 45:1. Only 0.19% of images exceed the 300-query budget.
- **Dead ends — do not revisit:** higher global Muon LR (0.01 NaN'd; flat even when QK-norm stabilized),
  bf16 autocast (didn't fix instability), QK-norm (TRT-undeployable, `qk_norm.md`), supervision density (3×),
  MAL standalone re-test (DEIM never ablates MAL without Dense O2O; our tie is consistent → withdrawn).

---

## Tier 1 — EXHAUSTED (all tried 2026-06-13, none promoted)

The only lever that moved this screen is **per-step optimization quality** (Muon → Adan). All other families
tied or regressed.

- **#1 PMC** — matcher class-cost ×((GIoU+1)/2)^0.5 (Stable-DINO, arXiv:2304.04742). 🔴 **tie**: mAP 0.2122
  (+0.0003 ≪ 0.003). D-FINE already weights geometry 7:2; CDN pre-empts churn. `exp/pmc` (`d392c80`).
  Lowers prior on the `cost_class 2→1` probe (#12).
- **#2 Cautious AdamW** — mask sign-disagreeing aux/AdamW updates (arXiv:2411.16085). 🔴 **tie**: mAP 0.2134
  (+0.0015), seed42 dipped. AdamW groups already well-conditioned; Muon acts on matrices the mask misses.
- **#3 Moonlight RMS-match** — rescale by 0.2·sqrt(max(A,B)) + muon_lr=base_lr (arXiv:2502.16982). 🔴
  **regression**: mAP 0.2091 (−0.0028). Cooler RMS-parity LR underperforms legacy base_lr×10. `muon_lr` knob
  kept (null→legacy).
- **#4 Muon-group WD** — decoupled λ (Moonlight Fig 2, arXiv:2502.16982). **λ=0.1** 🔴 regression (mAP 0.2057,
  −0.0062, over-regularized τ≈2k screen); **λ=0.03** 🔴 near-miss (mAP 0.2188 +0.0021, f1 0.568 +0.0045 vs Adan).
  Reverses the regression → **strongest §6 full-run candidate** (WD benefit grows with run length). `exp/muon-wd-003`.
- **#5 IA-BCE** — IoU-aware cls target t=s^α·u^(1−α) (Align-DETR, arXiv:2304.07527). 🔴 **regression**: mAP 0.2098
  (−0.0021), f1 0.5405 (−0.016). s² negatives run cls loss ~4-5× hot. Cls-target family exhausted.

---

## Tier 2 — profile-gated / fillers / contingent

- **#6 EMA observability + momentum bracket** — (a) log raw-vs-EMA each eval (observability, ~5 lines in
  `train.py:get_preds_and_gt`); (b) momentum bracket 0.9998→{0.999, 0.9999} (Karras, arXiv:2312.02696).
  (b) is run-length-specific → doesn't transfer to full runs. Skip unless investigating EMA dynamics.
- **#7 Pre-resized image cache** — mosaic decodes ~3.4 full-res (~2000×1500) JPEGs/sample only to squash to
  640 (`dataset.py:431-470`). **Measure first:** log data-wait vs step time; if <5%, skip. ETL script
  `src/etl/resize_cache.py` → long side 1280. Screen velocity only, not accuracy.
- **#8 Eval cadence** — val eval every epoch costs ~10-20% of the 60-min cap. Grid eval {4,8,12,16} then every
  epoch post-anneal; `train.eval_every: 1` default. **Measure wall-time first.** Accuracy-neutral.
- **#9 PreciseBN** — recompute BN stats post-cap on clean data (arXiv:2105.07576). 🔴 **tie/no-op**: mAP 0.2117,
  keep-if-better guard reverted both seeds. BN-gap falsified (HGNetv2-B0 has few BN layers + long EMA tracks eval).
- **#10 Backbone LR ratio** — 0.24→0.48 (RT-DETRv2, arXiv:2407.17140). 🔴 **tie**: mAP 0.2118 (−0.0001).
  Cold-backbone hypothesis doesn't pay off under the cap.
- **#11 Adan on aux groups** — 🟢 **PROMOTED** 2026-06-14 (arXiv:2208.06677, aux peak LR ×5). mAP **0.2167**
  (+0.0048 > margin), f1 **0.5635** (+0.0070), 2-seed std 0.0002/0.0005, latency-neutral. Now in baseline.
  `train.aux_optimizer: adan` + `train.adan_lr_mult: 5.0` (default off). Sha `4a09ba7`.
- **#12 Config-only probes** — (a) `cost_class: 2→1` (deprioritized: PMC #1 was inert); (b) `loss_ddf: 1.5→{0.75,
  3.0}` (never ablated anywhere in the DETR family). Exploratory; needs §6 full-run before adoption.

---

## Tier 3 — architecture & backbone (deep-dive 2026-06-13, user-requested)

**Meta-finding:** D-FINE-S is at a well-converged design point. Most arch upgrades are blocked by one of
**four gates**: (1) latency budget ≤1.05× TRT; (2) grid_sample TRT-fp16 footgun (qk_norm scar); (3) ImageNet-init
fairness (rule 2 — biggest backbone wins are self-supervised ViTs we can't test fairly); (4) already present in
the code (mixed-query-select, look-forward-twice, VFL-IoU targets, GFLv2-DGQP/LQE, reg_max=32). RT-DETRv2/v3
changed nothing in encoder/neck; RT-DETRv4 (Nov 2025) kept HGNetv2 and moved to KD. **Every Tier-3 idea changes
the inference graph except A7 (KD)** → TRT-row f1≈torch check mandatory, latency ≤1.05× must be measured.

Sources read: D-FINE 2410.13842, RT-DETR 2304.08069, RT-DETRv2 2407.17140, RT-DETRv3 2409.08475, RT-DETRv4
2510.25257, DEIM 2412.04234, DEIMv2 2509.20787, RF-DETR 2511.09554, LW-DETR 2406.03459, FasterNet 2303.03667,
LowFormer 2409.03460, PCN 2502.01303, StarNet 2403.19967, SPD-Conv 2208.03641, Rank-DETR 2310.08854, YOLOv9 2402.13616.

**Arch trio (2026-06-17) COMPLETE — 0/3 promoted.** Next remaining: **A7 KD** (top priority) or **A2** (backbone
probe), user-steered. Segment track (separate `task: segment` eval, off the detect screen): **A8**.

- **A7 KD from larger teacher** ⬜ **TOP PRIORITY** — RT-DETRv4 (arXiv:2510.25257): distill from vision-foundation-model
  teacher, keeping HGNetv2. **Train-only → graph byte-identical, latency 1.0, zero TRT risk** — clears every gate.
  `train.py` load frozen teacher (no-grad/fp16); `dfine_criterion.py` add logit-KD (KL/T) + optional feature/decoder
  L2 matching; `research_visdrone.yaml` `kd_teacher`/`kd_temperature`/`kd_weight`. ~50 LOC.
  ⚠️ **Teacher MUST be ImageNet-init** (rule 2). Fair recipe = 2-stage: train M/L ImageNet-init on VisDrone →
  distill into S. COCO teacher = product recipe, not screen candidate. ⚠️ **Pilot walltime first** (teacher
  forward/step costs student steps). Use L as teacher if M→S gap too small.
- **A1 SPD-Conv** 🔴 tie/slight-neg — space-to-depth in neck PAN SCDown (arXiv:2208.03641). mAP 0.2173 (+0.0006 ≪
  margin), params +0.387M, seed42 TRT-gap −0.004. YOLO small-object win didn't transfer to DETR neck.
  `exp/spd-conv` (`3be4729`). **Backbone-stem placement (b) remains open** if revisited.
- **A2 FasterNet-T1/T2** ⬜ — PConv backbone (CVPR'23, arXiv:2303.03667): ¼-channel dense conv cuts memory traffic,
  pure std-conv → TRT-trivial, GPU-honest. The one fair "can we beat HGNetv2?" probe. timm `features_only`,
  out_indices=(1,2,3), channels [128,256,512]; re-point mask-head stride-8 tap. Most likely **confirms B0 is
  near-optimal** (RT-DETRv4 re-affirmed Nov'25). High info value either way.
- **A3 RMSNorm + SwiGLU** 🔴 near-miss — DEIMv2 decoder modernization (arXiv:2509.20787): LayerNorm→RMSNorm,
  ReLU-MLP→SwiGLU. mAP 0.218 (+0.0013), TRT-clean, but sub-margin + **+0.788M params**. `exp/rmsnorm-swiglu`
  (`8dc49aa`). ⚠️ Keep `linear1`/`linear2` names (Muon group keys on them). **Open: RMSNorm-only (zero param cost)
  could be a free promotable change; also a TRT-clean stability brick (issue-#64).**
- **A4 Wide-tail decoder** ⬜ — D-FINE's dormant GO-LSD (`eval_idx`/`layer_scale` path already in
  `dfine_decoder.py` L489-495, OFF in all sizes). Train wide layers → distill into eval layer → strip at deploy
  (`convert_to_deploy` L528) → zero added inference ops. Needs from-scratch retrain. Speculative.
- **A5 Discrete cross-attn sampling** ⬜ — removes grid_sample (RT-DETRv2, already implemented as
  `cross_attn_method="discrete"`, `configs.py:24`/`arch/utils.py:219-256`). **Robustness, NOT accuracy**
  (RT-DETRv2: −0.5 mAP). Fix per-axis clamp bug `arch/utils.py:242` first (clamps both coords to h-1). Deploy only
  if grid_sample TRT regression recurs.
- **A6 Rank-DETR HMC** 🔴 regression — class cost × IoU^4 (arXiv:2310.08854). mAP 0.2066 (−0.0101), f1 0.54
  (−0.0235) — worst of arch trio. IoU^4 zeroes class cost below near-perfect IoU. **Class-cost-×-overlap family
  dead** (PMC ^0.5 tied, HMC ^4 regresses). `exp/hmc` (`bc8bbe8`).
- **A8 Finer mask-head** ⬜ — SEGMENT TRACK (off detect screen). Feed backbone 1/8 feature to `MaskDecoder`
  (currently 1/4 res → tiny objects get ~1px masks). Plumbing exists (`low_level_feat`/`mask_low_level_ch` in
  `dfine.py`); extend to s/m/l/x. ~30 LOC. Pairs with Boundary Loss (arXiv:1812.07032).

---

## Methodology fix — needs USER sign-off (frozen file, shifts all numbers)

`validator.py:54` caps mAP at 100 dets/image (torchmetrics default). 10.9% of VisDrone images have >100 GT
(p95 132, max 902). Dense-scene gains under-credited. Fix: `max_detection_thresholds=[1,100,500]` — but
`validator.py` is frozen (guide 1.3), shifts all numbers → needs approval + re-baseline.
**User decided 2026-06-13: leave as-is** (`validator.py` stays unmodified; this is informational only).
ETL hygiene (no sign-off): gray-fill VisDrone ignored-region classes at conversion time if source annotations exist.

## Product-recipe notes (outside the screen protocol — no run-queue slot)

- **Objects365 pretraining** for user fine-tuning: +2.2 AP on D-FINE-S (paper). Repo ships `obj2coco.pt` +
  `train.pretrained_dataset: obj2coco`. Campaign stays ImageNet-init; this is a README/config recommendation.
- **Zoom-crop ETL** (SAHI-style, arXiv:2202.06934): native-res 640² crops alongside full frames → +7.4 AP50
  VisDrone. Train-only, reversible. Product feature for drone/CCTV datasets.
- **Rare-class oversampling**: duplicate tail-class rows in `train.csv` (LVIS repeat-factor style). Cheap,
  dataset-specific — document, don't queue.
- **DINOv2 backbone for transfer**: RF-DETR (+2.0 AP) + DEIMv2 show few-shot transfer wins from self-supervised
  backbone, not arch. Product A/B: optional DINOv2-ViT-S tier (**Apache-2.0**; DINOv3 is non-commercial). Higher
  GPU latency (+58% for +2.4 AP). Keeps deformable grid_sample decoder → needs TRT-fp16 hardening.

## Excluded / answered — do not spend runs

| Idea | Verdict |
|---|---|
| DEIMv2 ViT/DINOv3 backbone (2509.20787) | Non-ImageNet → unfair; +58% latency for +2.4 AP; RoPE TRT-fragile; non-commercial. HGNetv2 DEIMv2-N is only +0.2 AP → ViT carries the gain. Decoder bits salvaged as A3. |
| RF-DETR DINOv2-ViT + projector (2511.09554) | Backbone-driven, non-ImageNet → unfair; still grid_sample decoder. Projector redundant with native multi-scale HGNetv2. Kept as product note. |
| More AIFI levels (stride-16+) | RT-DETR Table 3 rejects: 16× FLOPs, blows latency. v2/v3/D-FINE all keep `use_encoder_idx=[2]`. |
| DySample/FreqFusion/Gold-YOLO neck | All TRT-unsafe: grid_sample/CARAFE (DySample, FreqFusion) or MHSA in neck (Gold-YOLO). SPD-Conv (A1) is the safe alternative. |
| More decoder layers (3→4) | +0.3 AP / +0.4 ms + more grid_sample blocks. Worst risk/reward. Wide-tail (A4) is the free alternative. |
| Mobile backbones (MBv4, EfficientViT, RepViT, StarNet, etc.) | Depthwise/attention → GPU-bandwidth-bound, slower than HGNetv2 at equal AP despite lower FLOPs. FasterNet (A2) is the GPU-honest swap. |
| ConvNeXt-V2 nano | 7×7 depthwise = GPU-bandwidth tax. Accuracy fallback only. |
| RT-DETRv2/v3 encoder/neck changes | v2/v3 changed nothing there. RF-DETR projector solves a ViT-only problem we don't have. |
| Walltime LR cooldown / mosaic close | Removed 2026-06-13: screen-regime-only, no transfer. Fixed as methodology (epochs 100→30). |
| MAL standalone / Muon LR raise / bf16 | Withdrawn or answered (see Dead Ends above). |
| One-to-many aux supervision (Group-DETR, RT-DETRv3, Co-DETR, etc.) | Density axis, 3× falsified; +40-70% step cost eats gains at fixed walltime. |
| Dense O2O mixup / multi-scale training / more queries / NWD-RFLA-DotD / P2-DDQ-DQ-DETR-UAV-DETR | All blocked: dead lever, no DETR evidence, latency blowup, or variable-K TRT-hostile. |
| Schedule-Free AdamW / AdEMAMix / MARS / Prodigy / SOAP / Shampoo | Wrong regime: long-run or LLM-only; Prodigy underperforms tuned Adam on ViT; SOAP needs ≥8× Chinchilla data. |
| QK-clip / Muon momentum-Nesterov-NS retunes | Stability-only or Moonlight-ablated dead knobs. |
| GHM / PolyLoss / label smoothing / focal-γ retune | No DETR evidence. `train.label_smoothing` is a **no-op** (wired into unused `focal` loss, not VFL). |
| torch.compile + channels_last | Watchlist: plausible 10-30% speedup but engineering-heavy vs #7-8. Revisit if throughput plateaus. |
| Copy-paste augmentation | Zero DETR-family result; bbox-only paste is a downgrade. Density lever again. |

## Notes / constraints

- **Init policy:** ImageNet backbone only (rule 2). Obj365 recommendation is product-side.
- **TRT rule:** Tiers 1-2 are train-only (latency 1.0, zero export risk). Every Tier-3 idea changes the graph
  except A7 (KD) → TRT-row f1≈torch check + latency ≤1.05× measurement mandatory (A8 runs on the seg export).
- **Segment safety:** all Tier 1-2 ideas task-agnostic or optimizer-side (verify masks on promotion). Tier 3:
  A7/A2/A3/A4/A6 task-agnostic; A1 changes a downsample feeding mask tap; A5 changes shared deformable attn;
  A8 IS the segment change. A backbone swap (A2) must re-point the mask-head stride-8 tap to new channel count.
- **Top of queue:** §6 Adan COCO full-run (non-arch → fair) → §6 Adan+Muon-WD λ=0.03 full-run (the near-miss)
  → #6 EMA bracket or a Tier-3 arch pivot (user-steered).
