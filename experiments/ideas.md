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

**⚠️ Before the first candidate: the horizon-30 re-baseline is PENDING.** `train.epochs` changed
100→30 on 2026-06-13 (guide rule 9 + §8) — the old Muon numbers below were measured at horizon-100
and are no longer the bar. Re-run the control (`run_candidate.py`, 2 seeds, unchanged code) and let
`promote.py` re-pin `baseline.json`. This is also the natural moment to decide the **maxDets
validator fix** (§Methodology below) — one re-baseline covers both if approved.

Baseline to beat (horizon-100 history, to be replaced) = **Muon** (`baseline.json`): test mAP_50_95
**0.2061**, f1 **0.552**; margins = floor **0.003**. All Tier-1 ideas are train-only → expected
latency ratio 1.0 and **zero TRT-export risk** (the qk-norm lesson: any change that touches the
inference graph needs TRT-row validation — guide §3, `qk_norm.md`).

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

## Tier 1 — run in this order (after the horizon-30 re-baseline)

### 1. Position-modulated classification cost in the matcher (Stable-DINO PMC)
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

### 2. Cautious AdamW on the auxiliary groups (C-AdamW)
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

### 3. Moonlight update-RMS matching for the Muon group
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

### 4. Real weight decay on the Muon group
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

### 5. IoU-aware classification target swap — ONE slot: IA-BCE or GCL
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

### 9. PreciseBN: recompute BN statistics post-cap on clean-distribution data
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

### 10. Backbone LR ratio raise (config-only)
- **Paper:** RT-DETRv2 (arXiv:2407.17140) scales backbone LR by capacity: its **lightest backbone
  runs at ratio 1.0** to the head LR (R18 → 1e-4 = head LR); ours runs HGNetv2-B0 at ratio 0.24
  (6e-5 / 2.5e-4), a heavy-backbone value, with ImageNet-only init and a big domain gap.
- **Change (1 key):** `research_visdrone.yaml` `train.lrs.s.backbone_lr` override `0.00006 →
  0.00012` (ratio ~0.5); follow-up 0.0002 if it wins (the config comment already hints "up to
  0.0002"). No isolated published ablation (part of +1.4 aggregate) — honest flag.
- **Transfer:** ✅ lands as a better per-size default in the LR table (user-facing). **Risk:**
  backbone LR is a NaN amplifier historically — watch the NaN-recovery log. **Latency:** none.
  **Segment safety:** ✅.

### 11. Adan on the AdamW aux groups
- **Paper:** Adan (arXiv:2208.06677, TPAMI'24) — the only modern optimizer with a published
  DETR-family COCO win: Deformable-DETR-R50 50e **44.5 → 45.3** (+0.8 over tuned AdamW); Mask R-CNN
  +0.5 box/+0.5 mask; "half-epochs" claims (ViT-B 150e ≈ AdamW 300e).
- **Change (files):** vendored ~40-line Adan update in `muon.py`'s aux branch (3 buffers/param),
  gated by `train.aux_optimizer: adamw|adan`; aux-group LR ×5 as the starting point (their
  convention). Muon group untouched.
- **Transfer:** ✅ published gains are at full schedules. **Risk:** no Adan-vs-Muon head-to-head
  anywhere; LR/beta retune strains one-change-per-run; complexity rule (guide 1.6) bites unless it
  clearly wins. **Generality:** ✅ (seg-validated on Mask R-CNN). **Segment safety:** ✅ but it
  changes the mask-head optimizer — verify.
- **Run after** ideas 2–4 settle the optimizer picture.

### 12. Config-only probes (cheap landscape-mapping; one run each, lowest priority)
- **Matcher cost rebalance:** `configs.py:45` `cost_class: 2 → 1` — a genuine literature gap (every
  DETR since Deformable ships 2:5:2 unexamined); motivated by the same churn analysis as idea 1.
  Run only if idea 1 wins (raises this probe's prior) or as filler.
- **DDF weight bracket:** `configs.py` `loss_ddf: 1.5 → 0.75` (and separately `→ 3.0`) — nobody has
  ever ablated it (D-FINE paper, DEIM, DEIMv2 all keep 1/0.15/1.5/5/2 verbatim).
- **Transfer:** ⚠️ exploratory; a winning value still needs the §6 full-run sanity check before
  becoming a default. **Segment safety:** ✅.

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

## Excluded / answered — do not spend runs

| Idea | Verdict |
|---|---|
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
- **Re-rank rationale (notebook rule):** the notebook's standing pointer was "resume at Cautious" —
  the 2026-06-13 research pass put **PMC (#1)** ahead: the matcher cost is the one untouched
  quality-side surface, with two independent +0.4 results at 12-e schedules and a ~6-line diff;
  Cautious follows immediately at #2. Screen-regime ideas (cooldown, aug-close) were removed per
  user decision the same day; the horizon-30 constant replaces them as methodology.
- **Sequencing:** run the horizon-30 re-baseline first (decide maxDets at the same time). Ideas
  1–4 are mutually independent (safe to run in any order on the evolving trunk); 5 last in Tier 1
  (lowest posterior). Tier-2 #7/#8 need a 1-epoch profile before spending a slot.
- **Init policy unchanged:** ImageNet backbone only (guide rule 2). The obj2coco recommendation is
  product-side only.
- **Segment-safety summary (rule 10):** 2, 3, 4, 6–10 task-agnostic or optimizer-side (verify masks
  on promotion as usual); 1 shared-matcher (verify); 5 detect-only override; 11 changes the
  mask-head optimizer (seg-validated in its paper, verify).
- **TRT rule:** nothing in Tiers 1–2 touches the inference graph; if any future idea does, the
  TRT-row f1 ≈ torch f1 check (guide §3) is mandatory before trusting it.
