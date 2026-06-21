# Idea backlog — candidate improvements (researched 2026-06-13)

**Proposed run queue** for the autoresearch loop. A fresh agent: read this + `lab_notebook.md`
(Current state), then **run Tier 1 in order, one experiment at a time**, following
`EXPERIMENT_GUIDE.md` §5 (branch from `main_exp` → single change → `make test` → background
`run_candidate.py` → `promote.py` → notebook → `notify.py`). **One change per experiment. Read the
paper before implementing.** The per-experiment approval gate (guide 1.8) still applies in
interactive mode. Move an idea into `lab_notebook.md` once tried.

**Mission (guide §0, updated 2026-06-13):** improve **D-FINE-seg generally** — VisDrone is only the
screen. Every idea must improve the model/training recipe *as users run it* (full schedules, full
convergence); changes that only optimize the 60-min screen regime are methodology, not candidates
(two such ideas were removed by user decision 2026-06-13 — see Excluded). Ideas are ranked by
*mechanism generality × expected value*; dataset-specific tuning is demoted to §Product-recipe
notes / §Excluded.

**✅ Horizon-30 re-baseline DONE (2026-06-13, `baseline_h30`).** `train.epochs` changed 100→30 on
2026-06-13 (guide rule 9 + §8); the current-best Muon recipe was re-run unchanged at horizon-30, 2
seeds, and `promote.py` re-pinned `baseline.json`. The **maxDets validator fix was NOT applied**
(user decision 2026-06-13: leave detections-per-image as is — `validator.py` stays frozen/unmodified;
§Methodology below is informational only now, not a pending action).

Baseline to beat (current, horizon-30) = **Muon** (`baseline.json`, `baseline_h30`): test mAP_50_95
**0.2119**, f1 **0.5565**; margins = floor **0.003**. (Horizon-100 history was 0.2061/0.552.) All
Tier-1 ideas are train-only → expected latency ratio 1.0 and **zero TRT-export risk** (the qk-norm
lesson: any change that touches the inference graph needs TRT-row validation — guide §3, `qk_norm.md`).

**🆕 Tier 3 — architecture & backbone (deep-dive 2026-06-13, user-requested).** Now that the
train-only levers are exhausted, the new direction is **model architecture / a new backbone**. A full
literature pass (DEIM/DEIMv2, RT-DETRv2/v3/v4, RF-DETR, LW-DETR, the efficient-backbone zoo, small-object
necks) is written up as **Tier 3** below. Read its meta-finding first — most arch upgrades are blocked by
one of four gates and the candidates are narrow. **Unlike Tiers 1-2, every Tier-3 idea changes the
inference graph** → the TRT-row f1≈torch check (guide §3) is mandatory, the COCO-init full-run is biased
(judge on the ImageNet-init 2-seed screen, guide §6), and the approval gate (guide 1.8) is non-negotiable
before burning GPU. Sequencing is **user-steered**, not auto-queued ahead of Tier-2.

## Diagnosis driving this queue

Run shape (measured): ~809 steps/epoch (6,471 train imgs, bs 8) × ~21-22 epochs under the 60-min
cap ≈ **18k steps**. With the new 30-epoch horizon (2026-06-13): warmup 3 epochs, runs end ~65-80%
through the anneal (~8-30% of peak LR) — close to a converged run's end state, so screen deltas
should now track full-training deltas better. Two accepted regime constants: **mosaic never closes**
(pinned off — user decision; PreciseBN #9 is the cheap hedge if the mosaic-vs-clean BN gap matters)
and **EMA τ = 5,000 steps ≈ 6.3 epochs ≈ 28% of the run** (long; idea 6 probes it).

1. **Per-step gradient quality is the proven lever; supervision density is falsified** (Muon won;
   CDN, Dense O2O regressed; MAL tied). New quality-side territory, in order: the **matcher cost**
   (untouched so far), optimizer update shaping (cautious masking, per-shape RMS-match), and weight
   decay — currently **inert**: τ_wd = 1/(lr·λ) ≈ 1.6e7 steps vs an 18k-step run, ~1000× too weak
   to do anything.
2. **Transfer-to-full-training is the bar.** Every Tier-1 idea is a per-step mechanism that runs
   identically at any schedule length, with published evidence at 12-50-epoch schedules. The §6
   full-run confirmation remains the final check before adoption (screen deltas can still shrink at
   convergence — the timm ConvNeXt result saw Muon's lead evaporate by epoch 300).
3. **Walltime is fungible with accuracy** (screen-velocity only, not model value): mosaic decodes
   ~3.4 full-res (~2000×1500) JPEGs per sample only to squash them to 640; eval runs every epoch
   inside the cap. Reclaimed minutes = more epochs per run and tighter stop-epoch jitter
   (ideas 7-8, measure first).
4. Dataset facts (computed from the lab-box train split, 2026-06-13): 343,205 boxes; mean 53 /
   median 42 / p95 132 / max 902 objects per image; **55.4% of objects < 16 px** (21.9% < 8 px) at
   640; class imbalance car:awning-tricycle ≈ 45:1. Only 0.19% of images exceed the 300-query
   budget (kills the "more queries" idea). Our 0.2061 is exactly RT-DETR-R18-class on this split.

Dead ends already answered — do **not** revisit: higher global Muon LR (0.01 NaN'd; flat even when
QK-norm stabilized it), bf16 autocast (tried, didn't fix instability), QK-norm (TRT-undeployable,
`qk_norm.md`), supervision density (3×), **MAL standalone re-test** (DEIM never ablates MAL without
Dense O2O — its +0.3/0.4 is on top of Dense O2O; our standalone tie is consistent with the paper,
so the old queue item is withdrawn).

---

## Tier 1 — 🔻 EXHAUSTED (all 5 tried 2026-06-13, none promoted)

**All Tier-1 ideas have been run; none beat the bar. `baseline_h30` (Muon, 0.2119/0.5565) holds.**
Verdicts (full reasoning per idea in `lab_notebook.md`):
- #1 PMC (matcher class-cost ×((GIoU+1)/2)^0.5) → 🔴 **tie** (+0.0003 mAP)
- #2 Cautious AdamW (mask aux/AdamW updates) → 🔴 **tie** (+0.0015 mAP / +0.0020 f1; seed42 dipped below baseline)
- #3 Moonlight update-RMS match (+ muon_lr→base_lr) → 🔴 **regression** (−0.0028 mAP); legacy ×10 Muon LR vindicated
- #4 Muon-group WD λ=0.1 → 🔴 **regression** (−0.0062 mAP); over-regularizes the ~18k-step screen (λ=0.03 / §6 full-run deferred)
- #5 IA-BCE (IoU-aware cls target) → 🔴 **regression** (−0.0021 mAP / −0.016 f1); un-α'd s² negatives run cls loss ~4-5× hot

**Mechanistic conclusion:** the only lever that has moved this screen is **Muon (per-step optimization
quality)**. **Update 2026-06-14: the optimizer axis is ALIVE — #11 Adan 🟢 PROMOTED** (new best
0.2167/0.5635, +0.0048 mAP / +0.0070 f1; second win after Muon). **Tier-2 results:** #10 backbone-LR →
🔴 tie; #9 PreciseBN → 🔴 tie/no-op (guard reverted both seeds); **#11 Adan → 🟢 PROMOTED.** Skip
#6(observability)/#6(b)(run-length-specific, doesn't transfer)/#7/#8 (screen-velocity-only methodology —
not model candidates per mission §0). **User-approved 3-experiment set DONE (2026-06-14): #9 PreciseBN
[🔴 tie] → #11 Adan [🟢 PROMOTED, new best] → Muon-WD λ=0.03 [🔴 positive near-miss +0.0021/+0.0045,
sub-margin, vs Adan].** Net: +1 promotion (Adan). Sections #1–#5, #9, #10, #11 done — do not re-run.
**Top of queue: §6 Adan COCO full-run (non-arch → fair), then §6 Adan+Muon-WD λ=0.03 full-run (the
near-miss), then #6 EMA bracket or a Tier-3 arch pivot (user-steered).**

### 1. Position-modulated classification cost in the matcher (Stable-DINO PMC)  — 🔴 TRIED, REJECTED (tie, 2026-06-13)
- **Result:** test mAP_50_95 0.2122±0.0014 vs 0.2119 (gain +0.0003 ≪ 0.003), f1 0.557 (+0.0005),
  latency-neutral. The 12-e COCO +0.4 AP didn't transfer — D-FINE's cost already weights geometry
  7:2 over class and CDN pre-empts the churn PMC targets. See lab notebook. Section kept below for
  reference / follow-up ladder context.
- **Paper:** "Detection Transformer with Stable Matching" (arXiv:2304.04742, ICCV'23): modulate the
  class probability inside the matching cost by overlap, `p ← p·((GIoU+1)/2)^0.5` — removes the
- **Paper:** "Detection Transformer with Stable Matching" (arXiv:2304.04742, ICCV'23): modulate the
  class probability inside the matching cost by overlap, `p ← p·((GIoU+1)/2)^0.5` — removes the
  "confident-but-misplaced query steals the GT" failure mode and the multi-path matching
  instability. +0.4 AP on top of PSL (DINO-R50, **12-epoch** schedule, 49.8→50.2). Independently
  corroborated: Rank-DETR's high-order cost `p·IoU^4` +0.4 (H-DETR-R50, 12e; arXiv:2310.08854).
- **Change (files):** `src/d_fine/matcher.py:150-170` — hoist the pairwise GIoU above the class
  cost (it's already computed for `cost_giou`), then before the focal pos/neg cost lines:
  `out_prob = out_prob[:, tgt_ids] * ((giou_pair + 1) / 2).clamp(0, 1).pow(0.5)`. `cost_giou`
  unchanged. CDN bypasses the matcher → denoising unaffected. ~6 lines.
- **Why:** the matcher cost is the one quality-side surface this campaign hasn't touched. Matching
  churn wastes early steps in *every* run (full schedules included — the paper's gains are at
  full-recipe COCO); on sub-16-px boxes (55% of this dataset) IoU is near-binary and L1 is
  numerically flat, so the score-driven class cost currently dominates pair ranking — the exact
  pathology PMC fixes. Zero step-time cost (matcher is `@torch.no_grad`).
- **Transfer to full training:** ✅ mechanism active every step at any length; published gains are
  on standard (non-truncated) schedules.
- **Latency:** none (train-only; export graph identical). **Complexity:** very low.
- **Expected:** +0.003–0.005 mAP_50_95 if the 12-e COCO numbers transfer. Watch early epochs: if
  mAP_50 stalls (all-pairs IoU≈0 crushes the class cost initially), soften exponent to 0.25.
  Follow-up ladder if it wins: 2, then 4.
- **Generality:** ✅ COCO-validated mechanism, model-agnostic across the DETR family.
- **Segment safety:** ⚠️ shared matcher (segment adds mask costs on top — untouched). Mechanically
  task-agnostic; verify masks don't regress before it becomes a shared default.

### 2. Cautious AdamW on the auxiliary groups (C-AdamW)  — 🔴 TRIED, REJECTED (tie, 2026-06-13)
> Result: test mAP 0.2134 (+0.0015), f1 0.5585 (+0.0020), both < 0.003 margin. AdamW groups already
> well-conditioned; Muon (the working lever) acts on enc/dec matrices the mask leaves untouched.
> C-Muon follow-up low-prior. See lab notebook. Section kept for reference; skip it.
- **Paper:** "Cautious Optimizers" (arXiv:2411.16085, NeurIPS'24): zero update coords whose sign
  disagrees with the live gradient, renormalize by mask density. Vision evidence: timm's
  independent replication (`rwightman/timm-optim-caution`): vit_wee mini-IN 71.23→**73.52**;
  paper's MAE-ViT pretrain eval-loss win. Fair-tuning caveat: "Fantastic Pretraining Optimizers"
  (arXiv:2509.02046) bounds the honest speedup family at <1.2×.
- **Change (files):** `src/d_fine/muon.py` `step()`, **AdamW branch only** (after `upd` at :84-86):
  `m = (upd * p.grad > 0).to(upd.dtype); m.div_(m.mean().clamp_(min=1e-3)); upd.mul_(m)`.
  Gate behind `train.cautious` (default False) → `MuonWithAuxAdam(param_groups, cautious=...)`,
  threaded via `dfine.py:build_optimizer` + `train.py`. ~6 lines, no new state.
- **Why:** "don't step where unsure" = strictly better per-step progress; covers the backbone +
  det head + norms (the groups Muon doesn't touch). The paper's theory covers Adam-like momentum —
  masking the AdamW groups is the theoretically clean variant.
- **Transfer to full training:** ✅ per-step mechanism, no schedule dependence in the paper's
  theory; paper/timm gains measured on full (50-300-epoch) runs.
- **Latency:** none. **Complexity:** very low.
- **Expected:** small–moderate. Follow-up (separate experiment, only if this wins): **C-Muon**, mask
  `momentum ∘ grad` **before** Newton-Schulz, exactly as the reference `c_muon.py` does (post-NS
  masking would destroy orthogonality; neither ordering is covered by the paper's theory — label it
  heuristic in the notebook).
- **Generality:** ✅ data/task-agnostic; default-off flag.
- **Segment safety:** ✅ optimizer-side only; mask head is in the AdamW groups → verify, but
  mechanism is task-agnostic.

### 3. Moonlight update-RMS matching for the Muon group  — 🔴 TRIED, REJECTED (regression, 2026-06-13)
> Result: test mAP 0.2091 (−0.0028), f1 0.554 (−0.0025). The cooler RMS-parity Muon LR underperforms
> the legacy base_lr×10 scaling — answers the "blind ×10" question: hot Muon LR is genuinely better
> here, not lucky. muon_lr knob kept (null→legacy). See lab notebook. Section kept for reference; skip it.
- **Paper:** "Muon is Scalable for LLM Training" / Moonlight (arXiv:2502.16982), Eq. 4: rescale the
  orthogonalized update by **`0.2·sqrt(max(A,B))`** so Muon's update RMS matches AdamW's (~0.2) for
  every matrix shape → "Muon can directly reuse the LR and WD tuned for AdamW".
- **Change (files):** `src/d_fine/muon.py:38` — replace
  `update *= max(1, grad.size(0) / grad.size(1)) ** 0.5` with
  `update *= 0.2 * max(grad.size(0), grad.size(1)) ** 0.5`; **and re-anchor the Muon LR to the
  AdamW LR**: expose `train.muon_lr: null` (null → legacy `base_lr*10`) read at `train.py:220`,
  set `muon_lr = base_lr` in `research_visdrone.yaml`. One conceptual change (Moonlight defines the
  rescale *with* the re-anchor).
- **Why (and why this isn't the dead LR raise):** today's scaling runs the Muon group ~3× *hotter*
  than AdamW-RMS parity (square 256×256 matrices: update-RMS·lr ≈ 3.1e-4 vs the AdamW groups'
  ≈1e-4) — which is exactly why the global raise to 0.01 NaN'd/flattened. RMS-match is **cooler
  overall and redistributed across shapes**: square attn projections ×0.2·16=3.2, but the wide FFN
  down-proj (256×1024) gets ×6.4 — a relative reallocation a global multiplier cannot express.
  Closes the notebook's "muon_lr is a blind ×10" open question with a principled answer.
- **Transfer to full training:** ✅ if it wins, it transfers as a better per-shape default for every
  model size (shape-wise LR transfer is the mechanism's selling point); also de-risks future LR
  tuning.
- **Latency:** none. **Complexity:** trivial (1 line + 1 config key).
- **Expected:** uncertain sign — the accidental 3×-hot setting may suit short runs; a full-run
  check (§6) is cheap since this is a non-arch change. Either way decisive: flat/worse vindicates
  and documents the legacy scaling; better → transferable default. Follow-up knob if too cold:
  muon_lr = base_lr×1.5–2 (still shape-correct).
- **Generality:** ✅ across shapes/sizes; no vision-published validation yet (flag in notebook).
- **Segment safety:** ✅ Muon group only; mask head stays AdamW.

### 4. Real weight decay on the Muon group  — 🔴 BOTH LEVELS TRIED (λ=0.1 regression 2026-06-13; λ=0.03 positive near-miss 2026-06-14)
> λ=0.1 (2026-06-13, Muon baseline): test mAP 0.2057 (−0.0062, 2× margin), f1 0.55 (−0.0065) —
> over-regularized the ~18k-step screen (τ≈2k). λ=0.03 (2026-06-14, **Adan baseline**, `muon-wd-003`):
> mAP **0.2188 (+0.0021, sub-margin)**, f1 **0.568 (+0.0045)** — both seeds cleanly above Adan, no NaN.
> **λ=0.03 reverses λ=0.1's regression → mechanism sound, near the screen sweet spot, just under the
> 0.003 margin.** Rejected on the screen (rule-bound) but the **strongest-motivated §6 full-run
> candidate** (WD's benefit grows with run length → the screen under-measures it; on top of Adan it may
> clear the bar). muon_weight_decay knob lives on `exp/muon-wd-003` (NOT trunk — the earlier "kept on
> trunk" note was wrong). Follow-up λ-sweep 0.02/0.05 is low-prior (sub-margin). See lab notebook
> 2026-06-14. Skip as a screen candidate; revisit at §6 full-run.
- **Paper:** Moonlight (arXiv:2502.16982) Fig. 2: vanilla Muon (no WD) converges faster early but
  weights grow and it ends **worse**; with decoupled λ=0.1 it wins. Kimi K2 (arXiv:2507.20534)
  attributes Muon attention-logit explosions to the same weight growth. Timescale rule
  (arXiv:2405.13698): τ_wd = 1/(η·λ); our λ=1.25e-4 at η≈5e-4 → τ ≈ 1.6e7 steps — **current WD is
  inert at any feasible run length** (even a 75-epoch full run is ~60k steps).
- **Change (files):** `src/d_fine/dfine.py:build_optimizer` — the Muon group dict gets
  `"weight_decay": cfg`-driven value; expose `train.muon_weight_decay: null` (null → global
  `train.weight_decay`). Research: **0.1** lead (τ ≈ 2k steps at peak LR — Moonlight's operating
  point), 0.03 down-check by hand if it loses. The decoupled `p.mul_(1 - lr*wd)` is already in
  `muon.py:76`.
- **Why:** quality (Moonlight's ablation) + **cause-side stability**: bounds the weight/attention-
  logit growth behind this repo's whole NaN arc (`qk_norm.md` §1) without touching the export
  graph — the deployable counterpart to QK-norm. Pairs naturally with idea 3 (Moonlight always runs
  rescale+WD together; here sequenced as two slots, combo as a follow-up).
- **Transfer to full training:** ✅ and asymmetric in our favor — WD's benefit *grows* with run
  length (Moonlight's "ends worse" shows in the back half), so the truncated screen under-measures
  it: a positive screen verdict is strong evidence; a flat one warrants the §6 full-run check
  before discarding.
- **Latency:** none. **Complexity:** trivial.
- **Expected:** small–moderate on the screen (see asymmetry above). Optional separate probe:
  AdamW-groups λ → 0.01–0.1 (same timescale logic).
- **Generality:** ✅ principled (timescale rule); also a robustness story for issue-#64-class users.
- **Segment safety:** ✅ Muon group only.

### 5. IoU-aware classification target swap — ONE slot: IA-BCE or GCL  — 🔴 TRIED (IA-BCE), REJECTED (regression, 2026-06-13)
> Pre-step matched-IoU histogram (baseline_h30, diagnostic since removed): median 0.73, only
> 6.2% <0.1 → picked **IA-BCE** over GCL. Result: test mAP 0.2098 (−0.0021), f1 0.5405 (−0.016, guard
> tripped hard even at val-optimal threshold). IA-BCE's un-α'd s² negatives run cls loss ~4-5× hot vs
> fixed bbox/giou → loss balance shifts off localization; 12e COCO +1.3 didn't transfer. GCL untested
> but whole cls-target family now low-prior (MAL tie + IA-BCE regress). See lab notebook. Skip it.
- **Papers:** Align-DETR (arXiv:2304.07527, BMVC'24): IA-BCE, positives `BCE(s, t)` with
  `t = s^α·u^(1−α)`, α=0.25, negatives `s²·BCE(s,0)` — **+1.3 vs VFL head-to-head** (DINO-R50 12e:
  VFL 48.7 → IA-BCE 50.0), AP_S +2.7–3.7 on DAB/DN-DETR. Rank-DETR (arXiv:2310.08854): GIoU-aware
  target `t = (GIoU+1)/2` (GCL) — +0.5 (H-DETR 12e); stays informative when matched tiny boxes have
  IoU = 0 (where VFL/MAL/IA-BCE targets collapse to ~0 and feed churn).
- **Pre-step (free, no run):** log a matched-IoU histogram for ~1 epoch (5 debug lines in
  `loss_labels_vfl`, or offline from an existing checkpoint). Large IoU≈0 mass among matches →
  pick **GCL**; otherwise **IA-BCE**.
- **Change (files):** `src/d_fine/dfine_criterion.py:loss_labels_vfl` — IA-BCE: target
  `t = (pred_score.detach()**0.25) * (ious**0.75)` at positive idx, weight = `target + s²·(1−target)`
  (pos weight 1, focal s² on negatives); GCL: `t = ((giou_diag.detach()+1)/2)`, keep VFL weighting.
  Keep the `loss_vfl` key so `weight_dict` is untouched. Wire as a **detect-only** override in
  `research_visdrone.yaml` (don't mutate `base_cfg`), like the MAL plan was. ~10 LOC.
- **Why:** same family as MAL (which tied → lowered posterior, hence last in Tier 1) but with the
  only published direct VFL comparison, the best small-object numbers in the family, and 12-e
  schedule evidence. The score operating point will shift — the guard already benches at
  val-optimal threshold, so that's handled.
- **Transfer to full training:** ✅ loss mechanism present every step; Align-DETR also reports
  larger-schedule wins (50 ep +0.7 class).
- **Latency:** none. **Complexity:** low.
- **Expected:** uncertain (family caveat); check per-class deltas in `extended_metrics.csv`, not
  just mAP — un-down-weighted positives can skew the car-dominated class balance.
- **Generality:** ✅ COCO-validated; mechanism dataset-agnostic.
- **Segment safety:** ⚠️ shared cls loss → keep as detect-only override; verify segment before any
  default change.

---

## Tier 2 — profile-gated / cheap fillers / contingent

### 6. EMA observability + momentum bracket
- **(a) Raw-vs-EMA logging** — ship silently with the re-baseline (observability, not a treatment):
  evaluate both models each eval epoch; answers "how much does the EMA hide/help?" on our task.
  `train.py:get_preds_and_gt` already takes the model via `self.ema_model.model`; ~5 lines.
- **(b) Momentum bracket** 0.9998 → {0.999, 0.9999} — one run each at most. τ currently 5k steps ≈
  28% of the screen run (Karras et al., arXiv:2312.02696: EMA length has a sharp optimum that
  scales with run length). With the horizon-30 anneal in place the "EMA as the only annealer"
  rationale is gone, so this is now a plain hyperparameter probe.
- **Transfer:** (a) ✅ pure observability; (b) ⚠️ the optimal *value* is run-length-specific — a
  winning screen value does NOT carry to full runs (re-derive there); what transfers is the
  τ-vs-run-length rule of thumb for the docs. **Segment:** ✅ task-agnostic.

### 7. Input-pipeline throughput: pre-resized image cache (measure first)
- **Evidence:** repo arithmetic — mosaic at p=0.8 decodes E[3.4] full-res (~2000×1500) JPEGs per
  sample (~10 MP ≈ 70–125 ms CPU each batch-element) only to squash every tile to 640×640
  (`dataset.py:431-470`). Under a walltime cap, dataloader stalls cost epochs; CPU contention is
  also the main source of the ±2-3-epoch stop jitter between seeds.
- **Measure first (no run burned):** log data-wait vs step time for 1 epoch (or watch GPU util).
  If data wait < 5% of step time on the lab box, skip and note it.
- **Change (files):** one-off ETL script (`src/etl/resize_cache.py`): resize long side → 1280 px
  (2× the 640 canvas bounds detail loss for zoom-ins), labels are normalized → copied verbatim;
  point `train.data_path` at the cached copy in `research_visdrone.yaml`. No loader change.
- **Transfer:** model-accuracy-neutral by design; value = screen velocity + jitter reduction, and
  as a documented prep recommendation for any user with large images (wall-clock, not accuracy).
- **Latency:** none. **Complexity:** low. **Segment safety:** ✅ (labels/polys unchanged).

### 8. Eval cadence (stop paying validation tax every epoch; measure first)
- **Evidence:** `train.py` runs a full val eval + dense-scene metric computation every epoch inside
  the 60-min cap; at even 30–60 s/epoch that's ~10–20% of the budget not spent training.
- **Change (files):** `train.py` — eval on a fixed grid (e.g. epochs {4, 8, 12, 16}) then every
  epoch from ~epoch 17 (where the best checkpoint lives, post-anneal), always once right before the
  cap; guard `save_model` accordingly. Config: `train.eval_every: 1` default (prod unchanged).
- **Measure first:** one log line for eval wall-time. **Transfer:** screen velocity + a useful
  product knob for users with big val sets; accuracy-neutral. **Complexity:** low.
  **Segment safety:** ✅.

### 9. PreciseBN: recompute BN statistics post-cap on clean-distribution data  — 🔴 TRIED, REJECTED (tie / no-op, 2026-06-14)
> Result: test mAP 0.2117 (−0.0002), f1 0.554 (−0.0025); both within margin. Implemented with a
> **keep-if-better guard** (eval val mAP before/after, revert if worse) — and it **reverted on BOTH seeds**
> (val mAP 0.2602→0.2571 / 0.2607→0.2565), so the candidate = baseline recipe. The mosaic→clean BN-gap
> hypothesis is **falsified** here: HGNetv2-B0 has few BN layers + the long EMA window (τ≈5k) already tracks
> the eval distribution, and a 200-batch recompute is a higher-variance estimate that *adds* noise. Guard
> worked as designed (zero regression). Code default-off, off-trunk. See lab notebook 2026-06-14. Skip it.
- **Paper:** Wu & Johnson, "Rethinking 'Batch' in BatchNorm" (arXiv:2105.07576): EMA running stats
  are worst when train/eval input distributions differ — and ours permanently do, since the
  campaign trains on 80%-mosaic collages and **never closes mosaic** (accepted constant), while
  eval is clean frames. Population stats over 10³–10⁴ samples fix it at ~0.5% cost.
- **Change (files):** `train.py`, after the walltime break: run ~200 forward-only train batches
  through **val-style transforms** (resize-only) on the EMA model with BN in cumulative-momentum
  mode; eval {EMA, EMA+PreciseBN}, save the winner. Runs *outside* the cap → ~free. ~20 lines,
  gated by `train.precise_bn: false`.
- **Why now:** with aug-close declined, this is the only mechanism addressing the mosaic→clean BN
  gap — and it's the cheap, principled one.
- **Transfer:** ✅ any BN-backbone detector trained with heavy aug benefits at any schedule length
  (full runs close mosaic at epoch `epochs-5`, shrinking but not eliminating the gap — the official
  D-FINE recipe's clean tail is only ~9% of training).
- **Expected:** +0.001–0.004, possibly nil. **Complexity:** low. **Segment safety:** ✅.

### 10. Backbone LR ratio raise (config-only)  — 🔴 TRIED, REJECTED (tie, 2026-06-13)
> Result: test mAP 0.2118 (−0.0001), f1 0.5575 (+0.0010), no NaN. Ratio 0.24→0.48 is neutral at the
> ~21-epoch screen — the cold-backbone hypothesis doesn't pay off under the cap. Follow-up ratio ~0.8
> not pursued (tie → low prior). See lab notebook. Section kept for reference; skip it.
- **Paper:** RT-DETRv2 (arXiv:2407.17140) scales backbone LR by capacity: its **lightest backbone
  runs at ratio 1.0** to the head LR (R18 → 1e-4 = head LR); ours runs HGNetv2-B0 at ratio 0.24
  (6e-5 / 2.5e-4), a heavy-backbone value, with ImageNet-only init and a big domain gap.
- **Change (1 key):** `research_visdrone.yaml` `train.lrs.s.backbone_lr` override `0.00006 →
  0.00012` (ratio ~0.5); follow-up 0.0002 if it wins (the config comment already hints "up to
  0.0002"). No isolated published ablation (part of +1.4 aggregate) — honest flag.
- **Transfer:** ✅ lands as a better per-size default in the LR table (user-facing). **Risk:**
  backbone LR is a NaN amplifier historically — watch the NaN-recovery log. **Latency:** none.
  **Segment safety:** ✅.

### 11. Adan on the AdamW aux groups  — 🟢 PROMOTED (2026-06-14) → now in the baseline, retired from queue
> **Promoted as the new current best (`baseline.json` → `adan`, sha `4a09ba7`).** Adan (arXiv:2208.06677)
> on the aux (non-Muon) groups, aux peak LR ×5: test mAP_50_95 **0.2167** (+0.0048 > margin), f1 **0.5635**
> (+0.0070 > margin), 2 seeds std 0.0002/0.0005, latency-neutral, no NaN. Second clean optimizer win after
> Muon (bigger than Muon's +0.0043). Gated by `train.aux_optimizer: adan` + `train.adan_lr_mult: 5.0`
> (default off in `config.yaml`). Follow-ups (not auto-run): §6 Adan COCO-init full-run (fair — non-arch);
> Muon-WD λ=0.03 on the Adan baseline (experiment 3, running); Adan LR-mult ×3/×8 sweep (low-prior retune
> after a clean win); verify masks on a segment release (Adan now drives the mask-head optimizer). See lab
> notebook 2026-06-14. Section retained for forensics; do not re-run as a candidate.

### 12. Config-only probes (cheap landscape-mapping; one run each, lowest priority)
- **Matcher cost rebalance:** `configs.py:45` `cost_class: 2 → 1` — a genuine literature gap (every
  DETR since Deformable ships 2:5:2 unexamined); motivated by the same churn analysis as idea 1.
  ⚠️ **Deprioritized 2026-06-13:** idea 1 (PMC, the other class-cost-shape change) was inert here, so
  this probe's prior dropped — only worth a slot as last-resort filler now.
- **DDF weight bracket:** `configs.py` `loss_ddf: 1.5 → 0.75` (and separately `→ 3.0`) — nobody has
  ever ablated it (D-FINE paper, DEIM, DEIMv2 all keep 1/0.15/1.5/5/2 verbatim).
- **Transfer:** ⚠️ exploratory; a winning value still needs the §6 full-run sanity check before
  becoming a default. **Segment safety:** ✅.

---

## Tier 3 — architecture & backbone (deep-dive 2026-06-13, user-requested)

**Meta-finding (read first — honest).** D-FINE-S sits at a *well-converged* design point. A deep pass
over the whole real-time-DETR line returns a sobering conclusion: most "obvious" arch/backbone upgrades
are blocked by one of **four gates** —
1. **Latency budget** (≤1.05× TRT, or ≤1.20× w/ 2× margin). The encoder + deformable decoder are the
   bulk of D-FINE-S's 2.1 ms; almost any added compute blows it.
2. **The grid_sample TRT-fp16 footgun** (qk_norm scar). The highest-AP small-object necks
   (DySample, FreqFusion) and the deformable decoder itself all live on this fragile op.
3. **ImageNet-init fairness** (rule 2). The biggest published backbone wins — DEIMv2's DINOv3-distilled
   ViT, RF-DETR's DINOv2 ViT — are *self-supervised/distilled, not ImageNet-classification* pretrained,
   so they cannot be tested fairly in-campaign (and carry GPU-latency + license costs).
4. **Already present.** DINO mixed-query-select (`learn_query_content`), look-forward-twice (FDR's
   cumulative `pred_corners_undetach`), VFL-with-IoU targets, GFLv2-DGQP (= our LQE), reg_max=32
   (paper-optimal) are all already in the code. Don't re-add them.

Corroborating the wall: **RT-DETRv2 and v3 changed NOTHING in the encoder/neck** (v2 only tweaked decoder
deformable-attn; v3's gain is train-only one-to-many — the density lever we've falsified 3×). **RT-DETRv4
(Nov 2025) kept HGNetv2** and improved via foundation-model distillation — the family itself stopped
touching this skeleton. So calibrate expectations: Tier 3 is mostly *"probe whether the converged point
can be beaten,"* not a basket of likely wins. The candidates below are the few that clear all four gates.
**All but one change the inference graph → TRT-row f1≈torch check mandatory; approval gate applies.** The
exception is **A7 (knowledge distillation)** — train-only, graph-identical, zero TRT/latency risk, and the
one Tier-3 item the RT-DETR family itself moved to (RT-DETRv4) → promoted to **top priority**.

Sources read (verified from paper tables / repo source): D-FINE 2410.13842, RT-DETR 2304.08069 (Table 3
ablation), RT-DETRv2 2407.17140, RT-DETRv3 2409.08475, RT-DETRv4 2510.25257, DEIM 2412.04234, DEIMv2
2509.20787, RF-DETR 2511.09554, LW-DETR 2406.03459, FasterNet 2303.03667, LowFormer 2409.03460, PCN
2502.01303, StarNet 2403.19967, SPD-Conv 2208.03641, Rank-DETR 2310.08854, YOLOv9/GELAN 2402.13616.

**Ranked run queue (user-steered).** **A7 KD (train-only, top priority)** → ~~A1~~ 🔴 (tried 2026-06-17,
tie) → ~~A3~~ 🔴 (tried 2026-06-17, positive near-miss; RMSNorm-only ablation open) → A2 (the one fair
backbone probe) → A4 (free-at-deploy, speculative) → A5 (robustness, not accuracy) → ~~A6~~ 🔴 (tried
2026-06-17, regression). **Next remaining: A7 KD (top priority, needs a fair ImageNet-init teacher) or A2
(backbone probe), user-steered.**
**Autonomous arch trio (2026-06-17) COMPLETE — 0/3 promoted: ① A1 🔴 → ② A3 🔴 (near-miss) → ③ A6 🔴.
Adan 0.2167/0.5635 holds. Light arch levers don't beat the converged point (Tier-3 meta-finding confirmed).** **Segment track (separate `task: segment` eval,
later — NOT on the detect screen): A8 finer mask-head.** Read the paper, measure latency, run the TRT-row
check. (A7 and A8 were merged in 2026-06-13 from a review of an alternate research pass — user decision.)

### A7. Knowledge distillation from a larger teacher (TRAIN-ONLY — TOP PRIORITY) — ⬜ next
- **Paper/source:** RT-DETRv4 (arXiv:2510.25257, Nov 2025) — the family's own latest move: it improves the
  *small* RT-DETR via **distillation from a vision-foundation-model teacher**, keeping HGNetv2. Generic DETR
  KD = teacher soft-target logits (KL at temperature) + optional intermediate feature / decoder-output
  matching. (The alternate-research pass cited "KD-DETR arXiv:2105.07446" — unverified, don't rely on that
  ID; RT-DETRv4 is the solid, family-specific precedent.)
- **Why it leads Tier 3:** **train-only → the student IS the deployed model, inference graph byte-identical
  → latency 1.0, zero TRT-export risk.** It clears every Tier-3 gate (the only one that does), and KD is one
  of the most reliable accuracy levers in detection. It is also exactly where the RT-DETR line went next.
- **Change (files):** `train.py` — load a **frozen** teacher, run its forward in eval/no-grad each step;
  `dfine_criterion.py` — add a logit-KD term (KL on class logits ÷ T) + optional feature/decoder-output L2
  matching, weighted into the loss; `research_visdrone.yaml` — `train.kd_teacher: null`,
  `train.kd_temperature`, `train.kd_weight`. ~50 LOC.
- **⚠️ FAIRNESS (critical — do NOT shortcut):** the teacher MUST be **ImageNet-init** (rule 2).
  **Distilling from `dfine_m_coco.pt`/`dfine_l_coco.pt` leaks COCO knowledge into the student — the exact
  bias rule 2 forbids.** Fair recipe is 2-stage: (1) train a D-FINE-M/L **ImageNet-init** on VisDrone,
  (2) distill into the ImageNet-init S. A COCO-teacher variant is a **product recipe** (like obj2coco), not
  a clean screen candidate — flag it as such if run.
- **⚠️ Walltime caveat (under the 60-min cap):** a teacher forward per step costs time → fewer student
  steps (the failure mode that ate the one-to-many density ideas). M (~19M) over S (~10M) is a **modest**
  teacher gap → expect the lower end; use **L** as teacher if the M→S gap is too small. Cheap version:
  frozen teacher at fp16/no-grad. Caching teacher outputs is usually blocked by mosaic randomness — note it.
  **Pilot the walltime before committing** (one short run logging step-time delta).
- **Transfer:** ✅ general; RT-DETRv4 confirms the family direction. Published gains are full-schedule.
  **Latency:** 1.0 (train-only). **TRT risk:** none (graph identical). **Complexity:** medium (teacher
  load + loss wiring + producing a *fair* teacher is the real cost). **Segment safety:** ✅ KD applies to
  mask predictions too. **Expected:** highest-EV Tier-3 item, gated by (a) a fair teacher and (b) the
  walltime check — both must pass before trusting the screen number.

### A1. SPD-Conv detail-preserving downsampling (small/dense; best small-object evidence) — 🔴 TRIED, REJECTED (tie/slight-neg, 2026-06-17)
> Placement (a) tried (neck PAN SCDown → space-to-depth + 1x1). Result: test mAP_50_95 0.2173 (+0.0006 ≪
> margin), f1 0.5615 (−0.0020), avg_gain −0.0007, lat 1.0, **params +0.387M**, seed42 TRT-gap −0.004. The
> YOLO small-object win didn't transfer to the DETR neck: PAN stride 16/32 isn't where SPD's detail
> preservation pays, and RepNCSPELAN4 + deformable decoder already fuse multi-scale context. Rejected
> (simplicity: params↑ + faint Slice+Concat TRT-fragility for a tie). Code on exp/spd-conv (`3be4729`),
> off-trunk. **Placement (b) — backbone-stem space-to-depth — remains an open larger arch bet** if revisited.
> See lab notebook 2026-06-17. Section kept below for the (b) follow-up.
- **Paper:** Sunkara & Luo, "No More Strided Convolutions or Pooling" (SPD-Conv, arXiv:2208.03641,
  ECML-PKDD'22). Replace a *strided/pooled* downsample with **parameter-free space-to-depth** — slice the
  map into 4 stride-2 sub-maps, concat → 4×C at H/2×W/2 — then a stride-1 conv. **COCO Table 4 (verified):**
  YOLOv5-SPD-n 31.0 AP / **16.0 AP_S** vs 28.0 / 14.1; -s 40.0 / **23.5 AP_S** vs 37.4 / 21.1 → **~+2.6-3 AP,
  +11-13% AP_S, biggest on the small models / small objects** — exactly our regime (55% of boxes <16 px).
- **Change (files) — decide placement by measurement:**
  - **(a) ImageNet-clean (try first):** replace the neck's `SCDown` depthwise-stride-2 downsample in the
    PAN bottom-up path (`arch/hybrid_encoder.py` `downsample_convs`, used ~L391-408/482; `SCDown` in
    `arch/common.py`) with space-to-depth + 1×1-to-restore-channels. Backbone ImageNet weights untouched →
    fully fair under rule 2.
  - **(b) Higher-upside:** replace HGNetv2's early stride-2 downsample(s) (`arch/hgnetv2.py` `HG_Stage`).
    Preserves detail where it matters most (high-res early), but those conv weights then init random
    (ImageNet load is `strict=False` → tolerated, like the neck/head). Flag as a partial-init arch change.
- **Why:** strided downsampling *discards* the high-frequency detail tiny objects live on; space-to-depth
  moves it into channels instead of dropping it. General mechanism, parameter-free, Slice+Concat = same op
  class as YOLOv5 Focus.
- **Transfer:** ✅ structural, schedule-independent (YOLO evidence is full-schedule). ⚠️ evidence is
  *anchor-based YOLO, not DETR* — first DETR-family test, sign uncertain.
- **Latency:** the 4× channel inflation feeds a wider conv → net likely ~neutral but **MUST measure ≤1.05×
  before trusting.** **TRT risk:** low (Slice+Concat exports like Focus; no grid_sample) — but graph changes
  → TRT-row f1 check mandatory. **Complexity:** low-medium; needs retrain (non-strict load handles shapes).
- **Expected:** +AP_S if the YOLO numbers partially transfer; the most promising TRT-safe small-object arch
  lever. **Segment safety:** ✅ (downsampling change; mask head untouched) — verify on promotion.

### A2. FasterNet-T1/T2 backbone swap — the one GPU-honest "can we beat HGNetv2?" probe — ⬜
- **Paper:** Chen et al., FasterNet / PConv (CVPR'23, arXiv:2303.03667). **PConv** = take ¼ of channels,
  dense-conv only those, concat — cuts *memory traffic* (the real GPU bottleneck), not just FLOPs; pure
  standard conv, **no depthwise/attention → TRT-trivial.** IN-1k: T1 76.2 (B0 is 77.3), T2 78.9; T1 ≈4648
  fps V100 — faster-on-GPU than RepViT/EfficientViT/StarNet at equal accuracy.
- **Why this and not the mobile zoo:** triple-confirmed (LowFormer 2409.03460, FasterNet, PCN 2502.01303)
  that depthwise/attention "efficient" backbones (MobileNetV4, EfficientViT, RepViT, StarNet, GhostNetV3,
  EfficientFormerV2) are mobile/edge-fast but **GPU/TRT-bandwidth-bound → slower than HGNetv2 at equal AP
  despite lower FLOPs.** FasterNet is the *only* GPU-honest family with ImageNet `timm` weights + clean
  multi-scale features.
- **Change (files):** replace `HGNetv2` with a timm-backed backbone — `timm.create_model("fasternet_t1",
  features_only=True, out_indices=(1,2,3), pretrained=True)` → strides 8/16/32, channels **[128,256,512]**;
  set encoder `in_channels=[128,256,512]` in `configs.py` (HybridEncoder 1×1-projects to hidden_dim anyway).
  Wire ImageNet load (timm `pretrained=True` bypasses `ensure_pretrained`'s HF path in `dfine.py`/`utils.py`).
  Re-point the mask-head stride-8 tap (128-ch). ~half-day wiring.
- **ImageNet-init:** ✅ timm `.in1k` weights — fair under rule 2 (T2 if you want acc at a latency cost).
- **Transfer/Latency/TRT:** lands as a better/worse per-size default; expect **~flat AP at ≤ baseline
  latency** (T1 slightly below B0 on IN-1k). TRT risk low (std conv); graph change → TRT-row check.
- **Honest expected:** most likely **confirms HGNetv2-B0 is near-optimal** (RT-DETRv4 re-affirmed it in
  Nov'25). Run it as the *definitive* "is the backbone the bottleneck?" answer — cheap, one experiment,
  high information value either way. **Segment safety:** ✅ verify mask tap (128-ch stride-8).

### A3. RMSNorm + SwiGLU decoder modernization (from DEIMv2) — 🔴 TRIED, REJECTED (positive near-miss, 2026-06-17)
> Result: test mAP_50_95 0.218 (+0.0013), f1 0.5645 (+0.0010), avg_gain +0.0011 < margin 0.003 — both up,
> latency-neutral, **export TRT-clean** (RMSNorm→ONNX→TRT fine, no qk-norm footgun), but sub-margin + **+0.788M
> params** (SwiGLU's doubled linear1) → simplicity-rule keep. Code on exp/rmsnorm-swiglu (`8dc49aa`), off-trunk.
> **Open follow-ups (each a separate experiment):** (a) ablate which half drove it — **RMSNorm-only (zero
> param cost)** vs SwiGLU-only; RMSNorm-only could be a free, simpler, promotable change. (b) param-matched
> SwiGLU (dim_feedforward ×⅔). RMSNorm is now a known TRT-clean stability brick (issue-#64). See lab notebook
> 2026-06-17. Section kept below for the ablation follow-up.
- **Paper:** DEIMv2 "Real-Time Object Detection Meets DINOv3" (arXiv:2509.20787). Its "efficient decoder"
  keeps D-FINE's MSDeformableAttention + FDR + LQE + CDN **verbatim** and only swaps decoder
  **LayerNorm→RMSNorm** and the **ReLU-MLP FFN→SwiGLU**.
- **Change (files):** `arch/dfine_decoder.py` `TransformerDecoderLayer` — `norm1`/`norm3` (L205/L222)
  LayerNorm→RMSNorm; `forward_ffn` (L233-234) → SwiGLU (`linear1` → 2×width gate+value, SiLU-gate,
  `linear2`). Keep the fp16 clamp (L256) and the `Gate`.
  - ⚠️ **Muon confound:** `_is_muon_param` (`dfine.py:128`) keys on names `linear1`/`linear2` — **keep those
    names** (or update `MUON_TOKENS`) or the new FFN matrices silently leave the Muon group and confound the
    run (one-change rule).
- **Why:** RMSNorm is cheaper and drops mean-subtraction (an **fp16-stability win** — may also help the
  issue-#64 NaN class); SwiGLU is a strictly-more-expressive FFN at ~equal cost. Now-standard upgrades
  DEIMv2 adopted wholesale.
- **DON'T also port DEIMv2's "shared decoder pos-embed"** (hoist `query_pos_head` out of the loop): our
  decoder recomputes it per layer from the **FDR-refined** ref points (`dfine_decoder.py` L486/L531) —
  hoisting decouples the positional signal from the refined box and fights the cascade; the saving (2
  small-MLP calls over 3 layers) is negligible. Skip that part.
- **Transfer:** ✅ graph mechanism, schedule-independent. **Latency:** ~neutral on TRT-fp16. **TRT risk:**
  low (RMSNorm/SiLU-GLU fuse cleanly; no new grid_sample) — graph change → TRT-row check. **Complexity:**
  low (one module). **Expected:** small AP + possible stability bonus. **Segment safety:** ✅ shared
  decoder; verify masks.

### A4. Wide-tail decoder: train wide / eval narrow (dormant D-FINE machinery) — ⬜ speculative
- **Source:** D-FINE's own GO-LSD design; the `layer_scale`/`eval_idx` path **already exists**
  (`arch/dfine_decoder.py` L489-495) but is OFF in every released size (`eval_idx=-1`, `layer_scale=1`).
- **Mechanism:** set `eval_idx` < `num_layers-1` and `layer_scale`>1 so extra **wide** layers train *after*
  the eval layer and distill into it (GO-LSD), then are **stripped at deploy** (`convert_to_deploy` /
  `break` at L528) → **zero added inference ops / zero added grid_sample.** Buys decoder capacity for free
  at inference.
- **Why:** the one way to add decoder capacity **without** paying latency or adding a grid_sample block
  (unlike literal extra layers — RT-DETR Table 5: +0.3 AP / +0.4 ms for 3→4 AND linearly more of the fp16
  footgun).
- **Transfer:** ✅ if it wins it's a free-at-deploy default for every size. ⚠️ unused upstream, unmeasured
  at S scale, needs a **from-scratch retrain** (not a warm-ckpt flip), and the wide layers cost screen
  walltime (fewer epochs under the cap) → a screen tie could be a walltime artifact, not a real null.
  **Latency:** deploy-neutral (train-time cost only). **TRT risk:** low at deploy (wide layers absent from
  the engine) — verify the strip path exports clean. **Complexity:** medium. **Segment safety:** ✅.
  **Expected:** uncertain; architecturally the cleanest "more capacity, same latency" lever.

### A5. discrete cross-attn sampling — REMOVE grid_sample (robustness, NOT accuracy) — ⬜
- **Source:** RT-DETRv2 (2407.17140) discrete sampling; **already implemented here** as
  `cross_attn_method="discrete"` (config knob `configs.py:24`; impl `arch/utils.py:219-256`). Rounds
  sampling coords to integer pixels + gather → the exported graph has **no GridSample** (Gather/GatherND
  instead — the far-better-supported op family).
- **Why it matters:** directly removes the single op behind the qk_norm TRT-fp16 disaster (grid_sample
  fusion bug → deployed f1 0.0). The deployable endgame of the whole "TRT-clean" arc.
- **Cost:** RT-DETRv2 Table 4 = **−0.5 mAP_50_95** (likely worse on small/dense; mask hit ≥ box) → will
  probably **FAIL the accuracy bar.** So this is **not an accuracy candidate** — log it as a
  robustness/future-proofing option to deploy *only if* a real grid_sample/TRT regression bites again.
- **Bug to fix first (files):** `arch/utils.py:242` clamps **both** coords to `h-1` (`# FIX ME? for
  rectangle input`); HF's fix clamps x→`w-1`, y→`h-1`. We run **non-square letterboxed** inputs → the bug
  is live; port the per-axis clamp before trusting any discrete run.
- **Transfer/Complexity/Segment:** ✅ / low (flag flip + bug fix + short fine-tune from `model.pt`, freeze
  the offset predictor since rounding is non-diff) / ✅. **TRT:** the rare graph change that *reduces*
  fragility — still run the TRT-row fp16≈fp32 check.

### A6. Rank-DETR high-order matching cost (HMC) — 🔴 TRIED, REJECTED (regression, 2026-06-17)
> Result (α=4): test mAP_50_95 0.2066 (−0.0101), f1 0.54 (−0.0235), both past 2× margin — worst of the arch
> trio. IoU^4 is far too steep: it zeroes the class cost below near-perfect IoU, so early on (low IoU
> everywhere) the matcher loses its class signal and assigns ~purely on bbox/giou — the churn PMC reduces,
> amplified. **Class-cost-×-overlap family now dead here** (PMC ^0.5 tied, HMC ^4 regresses); a milder α=1-2
> might recover toward the PMC tie but not beat it → no slot. Code on exp/hmc (`bc8bbe8`), off-trunk. See lab
> notebook 2026-06-17. Section kept below for reference.
- **Paper:** Rank-DETR (NeurIPS'23, arXiv:2310.08854): multiply the Hungarian **class cost by IoU^α**
  (α≈4) so matching favors jointly high-cls + well-localized queries. +0.6 mAP (AP75 +1.1) on
  H-DETR/DINO-R50 12e. **Train-only, zero inference/TRT cost.**
- **Change (files):** `matcher.py` after L166 — `iou,_ = box_iou(box_cxcywh_to_xyxy(out_bbox),
  box_cxcywh_to_xyxy(tgt_bbox))` (helper already in `arch/utils.py`), fold `iou.clamp(0).pow(α)` into
  `cost_class`/`C`. ~5 lines, α sweep.
- **Caveat:** **PMC-adjacent** (matcher class-cost reshape) and **PMC tied here** (#1: D-FINE already weights
  geometry 7:2 over class + CDN pre-empts churn) → **lower prior.** The difference is HMC's steep IoU^4 vs
  PMC's gentle ((GIoU+1)/2)^0.5. Belongs with Tier-2 train-only fillers, not really "architecture" — listed
  here because the decoder research surfaced it. **Transfer:** ✅. **Segment safety:** ✅ shared matcher, verify.

### A8. Finer mask-head feature (1/8 input) — SEGMENT TRACK, off the detect screen (later) — ⬜
- **Source:** YOLACT-style real-time seg taps the *finest* feature for masks; merged 2026-06-13 from the
  alternate-research review (their idea G). For *later* segmentation-quality work — **not** a detect-screen
  candidate.
- **Why:** `MaskDecoder` runs at 1/4 res; an 8 px object → a ~2×2 mask feature (≈1 pixel). Feeding the
  backbone **1/8** (or stem) feature lifts small-object mask fidelity — and 55% of our objects are <16 px,
  so that's exactly where segment mask mAP is lost.
- **Change (files):** `arch/dfine_decoder.py` `MaskDecoder` accept + fuse a finer feature; the **plumbing
  already exists** — the nano path passes a `low_level_feat` for this purpose (`dfine.py:44-50` +
  `mask_low_level_ch` in `build_model`); extend it to s/m/l/x. ~30 LOC. **Latency:** low (mask decoder is
  tiny). **TRT risk:** low (bilinear-interp + conv, already in the mask path) — still run the TRT-row check
  on the **seg** export.
- **⚠️ Off the detect screen:** the campaign screens `task: detect` → this produces **zero** signal on the
  detect mAP/f1 `promote.py` reads; it **cannot be promoted/rejected by the harness.** Evaluate on a
  separate `task: segment` run (mask mAP_50_95). Hence "segment track / later," not the detect run queue.
- **Pairs with** a boundary-aware mask loss (Boundary Loss, arXiv:1812.07032 — small-object boundary
  precision) as a second segment-track item if mask quality is the goal.
- **Transfer:** ✅ any small-object segmentation user benefits. **Detect:** ✅ unaffected (mask head is
  task-gated — only built when `enable_mask_head`). **Complexity:** low-medium. **Segment safety:** it IS
  the segment improvement; verify detect is byte-unchanged (it is).

---

## Methodology fix — needs USER sign-off (frozen file, shifts all numbers)

**`validator.py:54` caps mAP at 100 detections/image** (torchmetrics default
`max_detection_thresholds=[1,10,100]`; line 57 even silences the warning). Measured on our split:
**10.9% of train images carry >100 GT objects** (p95 = 132, max 902), and the model emits up to 300
scored boxes — dense-scene gains are systematically under-credited, and best-ckpt selection
inherits the bias. The official VisDrone protocol evaluates up to 500 dets/image.
- **Fix:** `max_detection_thresholds=[1, 100, 500]` (one line) — but `validator.py` is **frozen**
  (guide 1.3) and every ledger number shifts → requires explicit user approval + a re-baseline.
  **The pending horizon-30 re-baseline is the natural moment: one re-baseline covers both.**
- **Why it matters beyond VisDrone:** maxDets=100 is correct COCO protocol but wrong for any dense
  dataset a user brings (crowds, aerial, retail shelves) — candidates that improve dense-scene
  recall are invisible to the current meter.
- Related ETL hygiene (no sign-off needed, do whenever): VisDrone GT classes 0 ("ignored regions")
  and 11 ("others") — if the YOLO conversion dropped them without masking pixels, dense unlabeled
  crowds train as hard negatives. Gray-fill ignored regions at conversion time if the source
  annotations are still around.

## Product-recipe notes (outside the screen protocol — no run-queue slot)

- **Default to Objects365 pretraining for user fine-tuning.** D-FINE paper (arXiv:2410.13842):
  Obj365→COCO pretrain is worth **+2.2 AP on D-FINE-S** (48.5→50.7) vs COCO-only; the repo already
  ships `dfine_s_obj2coco.pt` and `train.pretrained_dataset: obj2coco` plumbing. The *campaign*
  stays ImageNet-init (guide rule 2 — init must not bias arch comparisons), so this is a
  README/config-default recommendation + an optional manual full-run A/B (75e COCO-init vs
  obj2coco-init), not a screen candidate.
- **Zoom-crop ETL augmentation for high-res users.** SAHI fine-tuning (arXiv:2202.06934, corrected
  numbers): adding native-res 640² crops *alongside* full frames, full-image eval, gave TOOD
  +7.4 AP50 on VisDrone test-dev. Train-only, fully reversible (append crops to `train.csv`).
  Parked as a product feature for drone/CCTV-style datasets rather than a campaign idea: no paper
  ablates it at fixed-640 eval, and it's scale-distribution tuning, not a general mechanism.
- **Rare-class oversampling** (duplicate tail-class rows in `train.csv` at ETL time; LVIS
  repeat-factor style): cheap, but class-balance is dataset-specific — document, don't queue.
- **Self-supervised backbone (DINOv2/DINOv3) for custom-dataset transfer.** Both RF-DETR
  (arXiv:2511.09554 — own ablation: the DINOv2 backbone is the dominant lever, **+2.0 COCO AP** vs NAS's
  +0.3; **#1 on RF100-VL** custom-domain transfer) and DEIMv2 (DINOv3-distilled ViT-Tiny) show the real
  *few-shot transfer* win comes from a **self-supervised-pretrained backbone, not architecture** — exactly
  our production use case (users fine-tuning on small custom datasets). **Firewalled from the campaign**
  (rule 2 ImageNet-init keeps arch comparisons fair), but worth a product/roadmap A/B: an optional
  DINOv2-ViT-S backbone tier for "best accuracy on your own data," accepting **higher GPU latency** (ViT
  at uniform 1/16 is TRT-slower than HGNetv2 — DEIMv2-S is +58% latency for +2.4 AP) and licensing
  (DINOv3 is **non-commercial** → relevant for commercial use; **DINOv2 is Apache-2.0 — prefer it**). It
  keeps the deformable grid_sample decoder → needs the TRT-fp16 export hardening we already ship.

## Excluded / answered — do not spend runs

| Idea | Verdict |
|---|---|
| DEIMv2 ViT-Tiny/DINOv3 backbone + Spatial-Tuning-Adapter (arXiv 2509.20787) | **Non-ImageNet** (DINOv3-distilled) → unfair under rule 2; **+58% T4 latency for +2.4 AP**; RoPE is fp16/TRT-fragile (repo issues #151/#152 are TRT-fp16 correctness failures — same class as our qk_norm bug); DINOv3 is **non-commercial** (relevant for `agnify.ai`). The HGNetv2-backed DEIMv2-**N** path is only **+0.2 AP** over D-FINE-N → the ViT, not the decoder, carries the gain, and the ViT is exactly what we can't afford on GPU. The decoder bits (RMSNorm/SwiGLU) are salvaged as Tier-3 **A3**. |
| RF-DETR DINOv2-ViT backbone + MultiScaleProjector (arXiv 2511.09554) | Win is **backbone-driven** (own ablation: DINOv2 +2.0 AP, NAS only +0.3) and the backbone is DINOv2 **self-supervised, not ImageNet** → unfair under rule 2. Decoder is **still deformable grid_sample** (same TRT footgun retained — verified in `ms_deform_attn_func.py`). The projector solves a *ViT-has-no-multi-scale* problem we don't have (HGNetv2 is natively 8/16/32). DINOv2-for-transfer kept as a **product note**. Architecture (LW-DETR lineage) is *behind* D-FINE-X at matched size once the backbone is stripped. |
| More AIFI levels (`use_encoder_idx` beyond stride-32) | **Rejected by RT-DETR's own ablation** (2304.08069 Table 3: S5-only **46.8 AP / 7.9 ms** beats all-scales **46.4 / 12.2** on *both* axes); stride-16 AIFI ≈ **16×** the stride-32 attention FLOPs → blows the latency budget; v2/v3/D-FINE all keep `use_encoder_idx=[2]`. |
| DySample / FreqFusion / Gold-YOLO GD neck | The highest small-object-AP necks, all **TRT-unsafe**: DySample (2308.15085) + FreqFusion (2408.12879) use **grid_sample / CARAFE** (re-trigger the fp16 GridSample bug); Gold-YOLO (2309.11331) puts **MHSA in the neck** (attention-fuses-badly + heavy gather). SPD-Conv (Tier-3 A1) is the TRT-safe small-object alternative. |
| More decoder layers (`num_layers` 3→4+) | RT-DETR Table 5: only **+0.3 AP / +0.4 ms** for 3→4, AND each layer adds another MSDeformableAttention **grid_sample** block → linearly more of the exact fp16 footgun. Worst risk/reward. The free-at-deploy alternative is wide-tail (Tier-3 **A4**). |
| Mobile/edge backbones (MobileNetV4, EfficientViT, RepViT, StarNet, GhostNetV3, EfficientFormerV2) | Depthwise/attention-heavy → **GPU/TRT bandwidth-bound, slower than HGNetv2 at equal AP** despite lower FLOPs (confirmed by LowFormer 2409.03460, FasterNet 2303.03667, PCN 2502.01303). Speed wins are CPU/mobile/edge-only — wrong target. **FasterNet (PConv, pure std-conv)** is the one GPU-honest swap → Tier-3 **A2**. |
| ConvNeXt-V2 nano (atto/femto/pico) | Clean `features_only` wiring + good IN-1k acc, but **7×7 depthwise = the GPU-bandwidth tax** → slower than B0 at equal acc; only clears the bar under the ≤1.20×/2×-acc exception a tiny detector won't deliver. Accuracy *fallback* only, not a promotion candidate. |
| RF-DETR MultiScaleProjector / RT-DETRv2-v3 encoder changes | v2/v3 changed **nothing** in the encoder/neck (v2 = decoder-only deformable tweak; v3 = train-only one-to-many, our thrice-falsified density lever). RF-DETR's projector builds a pyramid *from a single ViT scale* — redundant with our native-multi-scale HGNetv2 + RepNCSPELAN4 neck (which YOLOv9's GELAN ablation 2402.13616 shows is already the optimal fusion block). |
| Stable-DINO PMC (matcher class-cost ×((GIoU+1)/2)^0.5, was Tier-1 #1) | **Tried 2026-06-13 → tie** (test mAP 0.2122 vs 0.2119, +0.0003 ≪ margin; f1 +0.0005). 12-e COCO +0.4 AP didn't transfer: D-FINE already weights geometry 7:2 over class in the cost and CDN pre-empts the churn. Code on `exp/pmc` (`d392c80`), off-trunk. Lowers the prior on the `cost_class 2→1` config probe (§12). |
| Walltime-triggered LR cooldown (was Tier-1 #1, 2026-06-13 draft) | **Removed by user decision 2026-06-13**: screen-regime-only — gated on `max_walltime_min`, literally inactive in full runs → no transfer. Real improvements must show in the standard setup. The legitimate core (screen ends near-peak-LR) was fixed as *methodology* instead: schedule horizon `epochs` 100→30 (guide rule 9/§8, re-baseline pending). |
| Walltime-fraction mosaic close (was Tier-1 #2) | **Removed by user decision 2026-06-13**: full runs already close mosaic (epoch `epochs-5`); the screen not closing it is accepted as a campaign constant (now pinned deterministic via `no_mosaic_epochs: 0`). PreciseBN (#9) covers the BN-stats slice of the concern. |
| MAL standalone re-test on Muon | Withdrawn — DEIM has **zero** standalone-MAL evidence (+0.3/0.4 is on top of Dense O2O); our tie stands. Idea 5 takes the classification slot. |
| Muon peak-LR raise / sweep | Answered: 0.01 NaN'd (`muon-lr`); flat even when stabilized (qk-norm @0.01). Only the RMS-match reformulation (idea 3) remains live. |
| bf16 autocast | Tried by user pre-campaign — didn't fix instability; no accuracy angle. |
| One-to-many aux supervision (Group-DETR 2207.13085, RT-DETRv3 2409.08475 +1.6 @ fixed epochs, Co-DETR +71% train cost, MS-DETR, OD-DETR EMA-teacher +1.4 @ +15-25% step time) | Density axis, thrice-falsified here — and at fixed *walltime* the +40–70% step cost converts ~21 epochs into ~13–15, eating the claimed gains. Revisit only with a 10-epoch pilot proving the walltime math. |
| Dense O2O via mixup at DEIM's operating point (p=0.5, off at 50%) | Parked: genuinely different from our falsified mosaic-1.0-always-on, but still the dead lever; only worth a slot if Tier 1 empties. |
| Multi-scale training (`augs.multiscale_prob`, plumbed but 0.0) | Skip: no isolated AP evidence in the whole RT-DETR/D-FINE family (aggregate freebies only); costs steps under the cap; sub-640 scales shrink already-tiny objects; mosaic already jitters scale [0.5,1.5]. |
| More queries (300→500+) / eval-time query raise | Only 0.19% of train images exceed 300 GT; no VisDrone ablation exists; eval-K ≠ train-K is mechanically possible (queries are selected top-K, `dfine_decoder.py:894-897`) but OOD and unendorsed (RT-DETR #257). |
| NWD / RFLA / DotD tiny-object metrics | Gains live in dense-prior *label assignment* (anchor-based); no o2o-matching analogue with clean DETR evidence; NWD's own VisDrone delta is weak (38.0→38.5 AP50). |
| P2 level, DDQ, DQ-DETR, UAV-DETR/Drone-DETR modules | Latency: P2 +120% GFLOPs; DDQ +26% latency + in-graph NMS (TRT-fp16 risk); DQ-DETR variable-K hostile to static TRT; UAV-DETR +28% GFLOPs @400e. |
| Schedule-Free AdamW (2405.15682) | Right theory for budgeted training, but redundant with EMA + the horizon-30 fix, no Muon variant, BN-stat complications. |
| AdEMAMix / MARS / Prodigy / SOAP / Shampoo | Wrong regime: long-run or LLM-only evidence; Prodigy underperforms tuned Adam on ViT; SOAP wins only at ≥8× Chinchilla data ratios. |
| QK-clip (K2), Muon momentum/Nesterov/NS-step retunes | Stability-only or Moonlight-ablated dead knobs; stabilized-higher-LR is already proven flat here. |
| GHM / PolyLoss / label smoothing / focal-γ retune | No credible DETR evidence 2024–2026. Repo note: `train.label_smoothing` is currently a **no-op** (only wired into the unused `focal` loss, not VFL) — don't tune it expecting effects. |
| torch.compile (regional) + channels_last | Watchlist, not dead: plausible 10–30% train speedup (backbone-only compile dodges the dynamic-shape recompile storm from variable CDN groups; Inductor cache amortizes across the campaign), but engineering-heavy vs ideas 7–8 which buy the same minutes surer. Revisit if throughput ideas plateau. |
| Copy-paste augmentation (Kisantal 1902.07296; AD-Det +0.6 VisDrone) | Parked: zero published DETR-family copy-paste result (we'd be first), bbox-only rect-paste is a downgrade of the published mask-based variants, and it's the density lever again. DEIMv2's Copy-Blend is the closest precedent if ever revisited. |

## Notes / constraints
- **Tier-1 post-mortem (2026-06-13):** all 5 ran, 0 promoted. The lever that *moved* this screen is
  per-step optimization quality (**Muon**); the matcher-cost (PMC), optimizer-update-shaping (Cautious,
  Moonlight, Muon-WD) and classification-target (MAL, IA-BCE) families are now each probed and exhausted.
  Two optimizer-side knobs were vindicated rather than improved (legacy Muon ×10 LR; the deliberately-inert
  global WD on this horizon). Net: the per-step-quality hypothesis is confirmed but Muon already captures
  most of the easy gain on it.
- **Sequencing (now in Tier-2):** Tier-1 exhausted. Skip #6(observability + run-length-specific
  momentum probe), #7, #8 (screen-velocity-only methodology — not model candidates per mission §0).
  **Tier-2 #10 (backbone-LR ratio raise) → 🔴 tie** (2026-06-13); **#9 PreciseBN → 🔴 tie/no-op**
  (2026-06-14, guard reverted both seeds); **#11 Adan → 🟢 PROMOTED** (2026-06-14, new best 0.2167/0.5635,
  +0.0048/+0.0070 — optimizer axis confirmed alive); **Muon-WD λ=0.03 → 🔴 positive near-miss**
  (2026-06-14, +0.0021/+0.0045 vs Adan, sub-margin — λ=0.03 reverses λ=0.1's regression, strongest §6
  full-run candidate). User-approved 3-experiment set DONE: PreciseBN [🔴] → Adan [🟢] → Muon-WD λ=0.03
  [🔴 near-miss]. **Top of queue: §6 Adan COCO full-run, then §6 Adan+Muon-WD λ=0.03 full-run, then #6
  EMA bracket or a Tier-3 arch pivot (user-steered).**
- **Init policy unchanged:** ImageNet backbone only (guide rule 2). The obj2coco recommendation is
  product-side only.
- **Segment-safety summary (rule 10):** 2, 3, 4, 6–10 task-agnostic or optimizer-side (verify masks
  on promotion as usual); 1 shared-matcher (verify); 5 detect-only override; 11 changes the
  mask-head optimizer (seg-validated in its paper, verify).
- **TRT rule:** nothing in Tiers 1–2 touches the inference graph (latency 1.0, zero export risk). **Every
  Tier-3 idea changes the graph EXCEPT A7 (KD — train-only, graph-identical, zero TRT/latency risk)** → for
  all the others the TRT-row f1 ≈ torch f1 check (guide §3) is mandatory and the latency ≤1.05× measurement
  is real, not assumed (A8 runs that check on the **seg** export).
- **Tier-3 post-mortem of the literature (2026-06-13, user-requested arch deep-dive):** the real-time-DETR
  family has *converged* on D-FINE-S's skeleton (RT-DETRv2/v3 changed nothing in encoder/neck; RT-DETRv4
  kept HGNetv2; the big backbone wins are non-ImageNet ViTs we can't test fairly). Net: arch gains are
  narrow and gated. Top arch bets, in order: **A7 KD** (train-only, zero TRT/latency risk, RT-DETRv4's own
  direction — top priority, modulo a *fair* ImageNet-init teacher + walltime check), **A1 SPD-Conv**
  (small-object, best evidence, TRT-safe), **A3 RMSNorm+SwiGLU** (cheap, low-risk), **A2 FasterNet-T1** (the
  one fair backbone probe — most likely confirms B0 is near-optimal). A4 (wide-tail) is speculative; A5
  (discrete sampling) is a robustness option not an accuracy play; A6 (HMC) is a PMC-adjacent train-only
  filler. **A8 (finer mask-head)** is a **segment-track** item (separate `task: segment` eval — off the
  detect screen). A7/A8 were merged 2026-06-13 from a review of an alternate research pass (user decision).
  **Sequencing is user-steered** — do NOT auto-run a Tier-3 arch change ahead of the Tier-2 train-only queue
  without approval (graph risk + biased COCO comparison + retrain cost).
- **Tier-3 segment-safety (rule 10):** A7 (KD) task-agnostic, applies to mask preds too; A2/A3/A4/A6
  task-agnostic (verify masks on promotion); A1 changes a downsample feeding the mask-head stride-8 tap
  (verify); A5 changes the shared deformable attn (verify); A8 IS the segment change (detect byte-unchanged).
  A backbone swap (A2) must re-point the mask-head stride-8 tap to the new channel count.
