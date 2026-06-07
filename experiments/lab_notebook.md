# Lab notebook — D-FINE-seg autoresearch

This file is the **memory across agents/sessions**. A fresh agent reads the *Current state* block
first to know exactly where to resume, then scans entries to avoid repeating dead ends. The
structured numbers live in `ledger.csv`; this file is the reasoning.

## Current state  (keep this block updated every iteration)
- **Baseline established:** yes — control, 3 seeds, sha `09d0463`. Do **not** re-train it.
- **Current best (`main_exp`):** baseline / control. Best *code* unchanged; methodology updated (see below).
- **Best metrics (test):** mAP_50_95 = 0.2018 , f1 = 0.5433 (margins 0.003 / 0.003; from `baseline.json`).
  Baseline's val-optimal threshold is 0.5, so the f1 metric switch (below) left `baseline.json` unchanged.
- **In progress:** idle.
- **Next idea:** DEIM **Dense O2O** (heavy mosaic) — the other half of DEIM; MAL (its loss half) was
  rejected standalone (below). Top of `ideas.md` Tier 1.
- **Notes for the next agent:**
  - **Methodology change (sha `6220c4c`, baked into trunk):** the f1 guard now benches at the
    **val-optimal conf threshold** (argmax-f1 on val, stored as `optimal_thresh` in
    `extended_metrics.csv`), not a fixed 0.5. Reason: the validator's old "optimal threshold" sweep was
    a no-op — `preds_postprocess` pre-filters at conf_thresh=0.5 before the validator, so the sweep
    never saw below 0.5 and always returned 0.5. Fixed to sweep the unfiltered `all_*` preds. `mAP_50_95`
    (primary) was never affected (it uses unfiltered scores). To eval an existing checkpoint without
    retraining: start training pointing at its folder and Ctrl+C during epoch 1 → the `finally` block
    evaluates the existing `model.pt` and writes `optimal_thresh`; then `make bench` reads it.
  - Three infra fixes baked in *before* the baseline — keep them: (1) `hgnetv2.py` dist-safe
    `get_rank`/`synchronize`; (2) `train.batch_size=8` pinned (auto `-1` OOMs on dense VisDrone); (3)
    mid-train CUDA OOM fails loudly in `train.py`. Also `train.epochs=100` (was 1000) sets the
    LR-schedule horizon — fixed constant (§8). Baseline mAP variance is tiny (std 0.0005) so the 0.003
    margin floor governs — a real win is very achievable.

---

Chronological log, newest first. One entry per candidate (promoted **or** rejected). Record the
*why*, not just the number — especially for failures.

Entry template:
```
## <date> — <name>   [PROMOTED | rejected | failed]
- Paper / source:
- Hypothesis:
- Change (files):
- Result (test, mean±std/seeds): mAP_50_95 <m>±<s> (best <b>), f1 <m>±<s>, lat ratio <r>, params <M>
- Read: why it worked / didn't. What it implies for the next idea.
```

---

<!-- entries below -->

## 2026-06-07 — mal (DEIM Matchability-Aware Loss)   [rejected — fair tie]
- Paper / source: DEIM, CVPR 2025 (arXiv:2412.04234). MAL = the loss half (Dense O2O is the other half,
  deferred to keep one change per experiment).
- Hypothesis: MAL keeps gradient on low-IoU matches (positive weight 1, target `iou^γ`) instead of
  near-ignoring them as VFL does (weight `iou`) → faster convergence, latency-neutral.
- Change (files): `dfine_criterion.py` (+`loss_labels_mal`, `mal_alpha`), `configs.py` (`loss_mal`,
  `losses=['mal',...]`, γ 2.0→1.5). exp/mal sha `349cb6b` (rebased on the methodology commit).
- Result (test, 3 seeds): mAP_50_95 0.2033±0.001 (gain +0.0015, **under** 0.003 margin), f1@val-optimal
  0.5393±0.0017 (gain −0.004, just **beyond** 0.003 margin), lat ratio 1.0, params 10.302M. 🔴 KEEP BEST.
- Read: **This experiment is why the methodology was fixed.** Under the old fixed-0.5 f1, MAL looked
  catastrophic (f1 0.4963, −0.047) — but MAL's γ=1.5 raises the positive target to a power, deliberately
  *suppressing* confidence scores, so its optimal operating point moved to **0.4** (consistent across all
  3 seeds; baseline stays 0.5). At its true threshold MAL's f1 recovers to 0.539 ≈ baseline's 0.5433 — a
  near-tie, not a regression. Conclusion: **MAL alone is ~neutral** here. That matches the paper — MAL is
  designed to manage the flood of low-quality matches introduced by **Dense O2O**; without it there's
  little for MAL to fix. Implication: try Dense O2O next; MAL likely only pays off *together* with it
  (worth re-testing the pair, but that's two changes — sequence Dense O2O first, then MAL+DenseO2O).

## 2026-06-07 — baseline   [PROMOTED — first baseline]
- Paper / source: n/a (unchanged control architecture).
- Hypothesis: establish the one-time control per EXPERIMENT_GUIDE §4.
- Change (files): none to the model. Infra fixes required to make the control runnable/comparable:
  `src/d_fine/arch/hgnetv2.py` (dist-safe pretrained-backbone load), `configs/research_visdrone.yaml`
  (`train.batch_size=8`, `train.epochs=100`), `src/dl/train.py` (fail loudly on mid-train CUDA OOM).
- Result (test, 3 seeds): mAP_50_95 0.2018±0.0005 (seeds .2025/.2016/.2012), f1 0.5433±0.0017,
  lat torch 13.57 ms / trt 2.1 ms, params 10.30M. All seeds hit walltime cap at epoch 22.
- Read: First two launch attempts produced a *degenerate* baseline (mAP 0.054±0.065, std > mean)
  from two compounding bugs — single-GPU `get_rank()` crash on ImageNet-backbone init, then a silent
  CUDA OOM (auto batch 11 on dense VisDrone batches) that the broad `except` in `train.py` swallowed,
  exporting 2-epoch models as "successful" runs. Fixed both, and separately found `epochs=1000`
  stretched the LR schedule so the ~22 real epochs never left warmup. Pinning `epochs=100` lifted
  mAP from 0.145 (old best single seed) to ~0.20 across *every* seed and collapsed variance to
  std 0.0005. Lesson for next agents: watch for silently-degraded runs; the OOM-loud guard now turns
  those into visible failures. Baseline is trustworthy; proceed to DEIM.
