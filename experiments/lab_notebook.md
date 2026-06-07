# Lab notebook — D-FINE-seg autoresearch

This file is the **memory across agents/sessions**. A fresh agent reads the *Current state* block
first to know exactly where to resume, then scans entries to avoid repeating dead ends. The
structured numbers live in `ledger.csv`; this file is the reasoning.

## Current state  (keep this block updated every iteration)
- **Baseline established:** yes — control, 3 seeds, sha `09d0463`. Do **not** re-train it.
- **Current best (`main_exp`):** baseline / control (`09d0463`).
- **Best metrics (test):** mAP_50_95 = 0.2018 , f1 = 0.5433 (margins 0.003 / 0.003; from `baseline.json`).
- **In progress:** idle.
- **Next idea:** DEIM (Dense O2O + matchability-aware loss) — top of `ideas.md` Tier 1.
- **Notes for the next agent:** Three infra fixes landed *before* the baseline and are now baked into
  the trunk — keep them: (1) `hgnetv2.py` uses dist-safe `get_rank`/`synchronize` (raw
  `torch.distributed` crashed single-GPU ImageNet-backbone init); (2) `train.batch_size=8` pinned in
  the research config (auto `-1` OOMs on dense VisDrone batches); (3) a mid-train CUDA OOM now fails
  loudly in `train.py` instead of exporting a half-trained model. Also `train.epochs=100` (was 1000)
  — it sets the LR-schedule horizon, treat as a fixed constant (§8). Baseline variance is tiny
  (std 0.0005 mAP) so the 0.003 margin floor governs — a real win is very achievable.

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
