# Lab notebook — D-FINE-seg autoresearch

This file is the **memory across agents/sessions**. A fresh agent reads the *Current state* block
first to know exactly where to resume, then scans entries to avoid repeating dead ends. The
structured numbers live in `ledger.csv`; this file is the reasoning.

## Current state  (keep this block updated every iteration)
- **Baseline established:** yes — control, 3 seeds, sha `09d0463`. Do **not** re-train it.
- **Current best (`main_exp`):** 🟢 **Muon** (sha `06f448e`, promoted 2026-06-08). Enc/dec attn+MLP
  matrices on Muon, rest (backbone/norms/biases/embeds/det+mask heads) on AdamW; gated by `train.use_muon`.
- **Best metrics (test):** mAP_50_95 = 0.2061 , f1 = 0.552 (margins 0.0006→floor 0.003 / 0.003; `baseline.json`).
  Beat control by +0.0043 mAP / +0.0087 f1, all 3 seeds, latency-neutral (trt 2.1ms, ratio 1.0).
- **In progress:** **full 75-epoch Muon confirmation run** (user-requested 2026-06-08) — COCO init +
  75 epochs + no walltime cap + use_muon, vs reference `det_s_2026-02-22` (test mAP_50_95 0.2316 / f1 0.5621).
  See `memory/project_muon_full_training.md`.
- **Next idea:** Tier-2 (loss/assignment) or re-test **MAL + Muon** (MAL was a standalone tie). Both
  supervision-density ideas (CDN #1, Dense O2O #2) were **rejected** — the cap bottleneck is per-step
  optimization, not positive count, which is why Muon (per-step efficiency) is the one that landed.
- **Notes for the next agent:**
  - **Methodology change (2026-06-08): 3 seeds → 2 (`harness.seeds=[42,123]`).** The screen is now
    2×60-min runs. No re-baseline: per-seed std (~0.0005–0.001) ≪ the 0.003 margin floor, so the floor
    governs promotion regardless of seed count; Muon's 3-seed baseline mean stays the bar. If a
    candidate's 2 seeds disagree by > margin, add a 3rd by hand. **Full/COCO runs are manual** and only
    an unbiased bar for *non-architecture* changes (COCO weights load identically) — for arch changes
    use shared-init full runs or defer COCO to real adoption. See EXPERIMENT_GUIDE §6 + rule 9.
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

## 2026-06-08 — muon (Muon optimizer for enc/dec 2D matrices)   [PROMOTED — first real win]
- Paper / source: Muon (Jordan et al., 2024) — Newton-Schulz-orthogonalized momentum for 2D weight
  matrices; ~faster convergence on speedruns. ideas.md #3.
- Hypothesis: after CDN (#1) and Dense O2O (#2) both *regressed* mAP, the lesson was that the
  walltime-cap bottleneck is per-step **optimization efficiency**, not supervision density. Muon
  attacks exactly that — orthogonalized updates on the high-condition-number enc/dec attention/MLP
  linears, where per-step gains compound most under ~22 epochs. Optimizer-only → zero inference latency.
- Change (files): new `src/d_fine/muon.py` (`MuonWithAuxAdam`, single-device, one `.step()` so the
  train loop is untouched); `dfine.py` `build_optimizer` gains a gated Muon path with an **allowlist**
  (`self_attn`/`cross_attn`/`linear1`/`linear2`/`gateway.gate`, ndim==2) so det/mask heads, embeddings,
  LQE, norms, biases can never leak in (verified: 25 matrices, 0 leaks); `train.py` passes the flag and
  gives the Muon group its own OneCycleLR peak (`base_lr*10*2`); `config.yaml` default `use_muon: False`;
  `research_visdrone.yaml` sets it true. exp/muon sha `06f448e`. Muon peak LR untuned (base_lr*10) —
  stable in all 3 seeds (no NaN recovery), so not divergent; possibly not optimal either.
- Result (test, 3 seeds): mAP_50_95 0.2061±0.0006 (gain **+0.0043**, > 0.003 margin, all seeds
  .2068/.2061/.2053), f1@val-optimal 0.5520±0.0022 (gain **+0.0087**, guard improved), lat trt 2.1ms
  (ratio 1.0), params 10.302M. 🟢 PROMOTE.
- Read: First change to beat the control on the **primary** metric beyond noise, with the guard *also*
  up and latency flat — and consistently across every seed (std 0.0006). Confirms the diagnosis from the
  two rejections: the lever that matters under the cap is optimizer efficiency. **Simplicity check:** it
  adds a ~90-line self-contained optimizer + a gated flag — non-trivial, but the win is clean, multi-seed,
  zero-latency, and the mechanism is general (not VisDrone-specific) so it should transfer to COCO; the
  added code is isolated and default-off. Net: complexity justified — promote. **Segment safety:** mask
  head + mask_decoder stay on AdamW (allowlist excludes `mask`), so the segment path is unaffected; Muon
  only touches the shared detection transformer. **Open:** Muon LR is a blind base_lr*10; a short sweep
  could yield more. Next: user-requested full 75-epoch confirmation vs the COCO-init Feb reference.

## 2026-06-08 — dense-o2o (DEIM Dense O2O / full mosaic)   [rejected — mAP regressed]
- Paper / source: DEIM, CVPR 2025 (arXiv:2412.04234). Dense O2O = pack more objects/image (full
  mosaic) → more O2O positives/step. ideas.md #2. (MAL is the loss half, rejected standalone 2026-06-07.)
- Hypothesis: full mosaic attacks O2O sparsity — the main convergence bottleneck under the 60-min cap —
  so denser supervision should lift mAP in ~22 epochs. Train-time aug → zero latency.
- Change (files): `configs/research_visdrone.yaml` `train.mosaic_augs.mosaic_prob` 0.8→1.0, as a
  **detect-only** override (mosaic degrades masks, CLAUDE.md #6 / GUIDE rule 10 — never in segment
  defaults). 1 line. exp/dense-o2o sha `198264e`. No OOM at batch_size=8 (peak VRAM ~95%, survived).
- Result (test, 3 seeds): mAP_50_95 0.1946±0.0011 (gain **−0.0072**, well past 2× margin — a real
  *regression*), f1@val-optimal 0.5450±0.0014 (gain +0.0017, within margin), lat trt 2.1ms (ratio 1.0),
  params 10.302M. 🔴 KEEP BEST.
- Read: mAP dropped clearly (−0.0072, std only 0.0011 → not noise) while f1 nudged *up* (+0.0017). The
  split is the tell: heavier mosaic makes a harder training distribution whose schedule (mosaic-close
  never reached under the cap, GUIDE §8) doesn't finish in ~22 epochs → localization/mAP suffers, but
  the denser positives slightly improve the classification operating point (f1). Net: more supervision
  density does *not* beat the harder distribution within the walltime budget here — same lesson as CDN
  (#1), from the opposite lever. Implication: the convergence bottleneck under the cap is **not** O2O
  positive-count; don't keep chasing supervision-density ideas (Group-DETR #4 likely same fate). The
  MAL+Dense O2O pairing is also unlikely to pay now (its base, Dense O2O, hurts mAP standalone). Pivot
  to the optimizer (Muon, #3): per-step *efficiency* rather than per-step *supervision*.
- Paper / source: Contrastive DeNoising, DINO (arXiv:2203.03605), inherited by RT-DETR/D-FINE. ideas.md #1.
- Hypothesis: dense VisDrone has large `max_gt_num`, so `num_group = num_denoising // max_gt_num`
  (`arch/utils.py:380`) floors to 1 — we run the *minimum* denoising. Raising `num_denoising` 100→300
  restores multiple noised-GT groups → denser, stable positives early when O2O is sparse. Train-only
  (`dfine_decoder.py:971` gates on `self.training`) → byte-identical export, zero latency cost.
- Change (files): `src/d_fine/configs.py:19` `num_denoising` 100→300 (1 line). exp/cdn-denoising sha `21970e4`.
- Result (test, 3 seeds): mAP_50_95 0.2004±0.0003 (gain **−0.0014**, below 0.003 margin — a slight
  *decrease*), f1@val-optimal 0.5403±0.0005 (gain −0.003, at margin edge), lat trt 2.1ms (ratio 1.0),
  params 10.302M. 🔴 KEEP BEST.
- Read: No win — mAP nudged *down*, not up, and variance is tiny (std 0.0003) so it's a real flat/slight-
  negative, not noise. Likely the extra dn tokens raised per-step cost enough to cost a fraction of an
  epoch under the 60-min cap, cancelling any denser-supervision benefit (the documented trade-off in
  ideas.md). The groups→1 starvation theory may also just not bind here: VisDrone's `max_gt_num` is so
  large that even 300 tokens still yields very few groups. Conclusion: CDN scaling alone is neutral-to-
  slightly-negative under the walltime cap; not worth the extra train cost. Implication: pursue the
  supervision-density gain through aug instead (Dense O2O, #2) rather than more dn tokens.

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
