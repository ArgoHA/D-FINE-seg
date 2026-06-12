# Idea backlog — candidate improvements

**Approved run queue** for the autoresearch loop. A fresh agent: read this + `lab_notebook.md`
(Current state), then **run the 5 ideas below in order, one experiment at a time**, following
`EXPERIMENT_GUIDE.md` §5 (branch from `main_exp` → single change → `make test` → detached-tmux
`run_candidate.py` → `promote.py` → update notebook → `notify.py`). **One change per experiment.
Read the paper before implementing.** All five are train-only / latency-neutral (expected ratio 1.0).
Move an idea into `lab_notebook.md` once tried. Per-idea fields: paper, change+files, why under the
cap, latency, complexity, expected, COCO transfer, segment safety (GUIDE rule 10).

Baseline to beat = **Muon** (`baseline.json`): test mAP_50_95 **0.2061**, f1 **0.552**; margins =
floor **0.003**. Diagnosis driving the queue: the binding lever under the 60-min / ~22-epoch cap is
**per-step optimization efficiency** (where Muon landed), **not** supervision density (CDN and Dense
O2O both regressed). 4 of 5 ideas tune the optimizer; MAL (#3) is the one loss-signal bet.

---

## Run queue (Tier 1 — approved, run in this order)

### 1. Muon peak-LR retune  ← run first (cheapest, highest EV)
- **Paper:** Muon (Jordan 2024); "Muon is Scalable for LLM Training" / Moonlight (arXiv:2502.16982).
- **Change (files):** `src/dl/train.py:220` sets `muon_lr = base_lr*10` (=0.0025); OneCycle then ×2
  at `:243` → the Muon group's **peak LR = 0.005**. Decouple from `base_lr`: expose `train.muon_lr`
  (default `base_lr*10` to preserve current behavior) and override in `research_visdrone.yaml`.
  **Lead experiment: peak 0.01** (`muon_lr = base_lr*20`). Follow-ups by hand if it wins: peak 0.015
  (`base_lr*30`), and down-check peak 0.0025 (`base_lr*5`). *Principled alternative (instead of a raw
  sweep):* Moonlight update-RMS match in `_muon_update` (`muon.py:33-38`) — rescale the orthogonalized
  update to AdamW's RMS ≈0.2, which removes the blind `*10` coupling and makes the LR transfer.
- **Why under the cap:** our peak (0.005) is ~4× below Muon's known-robust band (~0.01–0.02, where
  Muon's LR-transfer property holds). The enc/dec matrices are likely under-stepping in ~22 epochs;
  per-step optimization is the proven lever.
- **Latency:** none (train-only). **Complexity:** trivial (one scalar / one rescale).
- **Expected:** small–moderate mAP gain, or confirms `*10` was near-optimal (closes the notebook's
  open question either way).
- **COCO transfer:** ✅ strong — Muon LR transfer is the mechanism's selling point.
- **Segment safety:** ✅ touches only the Muon group's LR; mask head stays on AdamW.

### 2. Cautious optimizer (Cautious-Muon + C-AdamW)
- **Paper:** Cautious Optimizers, "Improving Training with One Line of Code" (arXiv:2411.16085, NeurIPS'24).
- **Change (files):** in `src/d_fine/muon.py` `step()`, after `upd` is computed in **both** branches
  (Muon `:74`, AdamW `:83-85`):
  `m=(upd*p.grad>0).to(upd.dtype); m/=m.mean().clamp_(min=1e-3); upd*=m`. Gate behind a `train.cautious`
  flag threaded via `dfine.py:build_optimizer` + `train.py`. ~4 lines, no new state/params.
- **Why under the cap:** zeroes update coords whose sign disagrees with the live gradient ("don't step
  if unsure") → strictly more progress per step; preserves Adam's Hamiltonian/convergence. Stacks on
  Muon instead of replacing it; the det head + backbone (on AdamW) benefit too.
- **Latency:** none (train-only). **Complexity:** very low (one elementwise mask).
- **Expected:** moderate convergence speed-up → mAP up in the truncated window (paper: 1.47× on Llama;
  consistent on ViT/MAE image tasks).
- **COCO transfer:** ✅ data/task-agnostic mechanism; validated on vision.
- **Segment safety:** ✅ optimizer-side mask only; no arch/mask-loss change. It is a shared default so
  it *will* run on segment (rule 10) — safe and likely helpful; verify masks don't regress.

### 3. MAL on the Muon baseline
- **Paper:** DEIM, CVPR'25 (arXiv:2412.04234). MAL = `-q^γ·log(p) − (1−q^γ)·log(1−p)` for positives,
  `−p^γ·log(1−p)` for negatives, **γ=1.5** (q = pred↔GT IoU).
- **Change (files):** re-apply the `exp/mal` diff onto today's Muon trunk (MAL is **not** on `main_exp`
  — it predates Muon): `dfine_criterion.py` (+`loss_labels_mal`, `mal_alpha`), `configs.py`
  (`losses=['mal','boxes','local']`, γ 2.0→1.5). Single change vs the Muon baseline.
- **Why under the cap:** MAL keeps gradient on low-IoU positives (target `q^γ`, weight 1) instead of
  VFL near-ignoring them (weight `iou`) — a per-step *gradient-signal* fix, not a density fix. Tests
  whether the standalone tie tips positive now that the optimizer is stronger.
- **Latency:** none. **Complexity:** low (loss already written on `exp/mal`).
- **Expected:** uncertain — **caveat:** the paper's +0.3–0.4 AP is measured *on top of Dense O2O*'s
  low-quality-match flood, which Muon does not create; standalone MAL was a tie here. Cheap, user-named.
- **COCO transfer:** ✅ DEIM is COCO-validated.
- **Segment safety:** ⚠️ swaps the *shared* classification loss → wire as a **detect-only** override in
  `research_visdrone.yaml` (do **not** mutate `base_cfg`), or verify segment masks don't regress (MAL
  shifts the score operating point).

### 4. EMA decay retune  ← cheap filler, low transfer
- **Paper:** EMA-of-weights / iterate-averaging (under the cap the LR barely decays, so the EMA *is*
  the smoothing — schedule-free intuition).
- **Change (1 key):** `research_visdrone.yaml` `train.ema_momentum` `0.9998`→`0.999` (ramp at
  `train.py:66` unchanged).
- **Why under the cap:** `0.9998` ≈ 5000-iter window on a ~17k-iter still-improving run; a faster EMA
  (~1000-iter) tracks the moving, near-peak-LR optimum better.
- **Latency:** none. **Complexity:** trivial.
- **Expected:** small mAP/f1 bump.
- **COCO transfer:** ⚠️ **walltime-specific** — a full COCO run wants high decay back; don't carry the
  value over. **Segment safety:** ✅ task-agnostic.

### 5. Adan on the AdamW aux groups
- **Paper:** Adan, "Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"
  (arXiv:2208.06677).
- **Change (files):** in `dfine.py:build_optimizer`, run the 4 non-Muon groups (backbone, enc/dec
  norms+biases, det+mask heads, rest) under Adan; Muon group unchanged. ~40-line custom optimizer (or
  an `adan-pytorch` dep), gated behind a flag.
- **Why under the cap:** Adan reaches comparable accuracy in ~half the epochs — directly the short-run
  regime. The det head *and* backbone are on AdamW, so the leverage is real.
- **Latency:** none (train-only). **Complexity:** **medium-high** — a second optimizer + 3 buffers/param;
  the simplicity rule (GUIDE 1.6) may bite unless it clearly wins.
- **Expected:** moderate; +0.5–1.2% AP over AdamW in the paper's COCO detection runs.
- **COCO transfer:** ✅ COCO-validated. **Segment safety:** ✅✅ uniquely validated on Mask R-CNN
  instance seg (+mask AP); still verify, since it changes the mask-head optimizer.

---

## Excluded / not queued
- **Group-DETR** (one-to-many auxiliary query groups, arXiv:2207.13085) — a supervision-**density**
  mechanism (K× positives), the exact lever CDN (#rejected) and Dense O2O (#rejected) both failed on;
  the notebook already predicts "likely same fate," and it is the costliest/most complex for segment
  (mask head runs over K× queries). Not worth a slot under the current diagnosis.

## Future directions (only if the queue is exhausted)
- **Tier 2 — loss / assignment:** quality-focal / IoU-aware refinements orthogonal to D-FINE's
  existing FDR/FGL; RankDETR-style rank-consistency losses. Matcher cost reweighting for small/dense
  objects (`matcher.weight_dict`, `configs.py:42`) is ⚠️ VisDrone-specific — flag for manual COCO check.
- **Tier 3 — architecture (watch latency + COCO-init bias):** encoder/neck fusion tweaks; activation/
  norm swaps; `query_select_method` alternatives (`configs.py:25`). Any inference-param/FLOP add must
  clear the *larger* margin AND justify complexity; judge arch ideas on the 2-seed ImageNet screen
  (COCO-init is biased for arch changes — GUIDE §6).

## Notes / constraints
- Init is **ImageNet backbone only** (constant); neck/head start random.
- **Segment safety summary (rule 10):** #1 Muon-LR and #4 EMA are fully segment-safe; #2 Cautious is
  safe but runs on segment as a shared default (verify); #3 MAL must be a detect-only override; #5 Adan
  changes the mask-head optimizer (seg-validated, but verify).
- **Run-order rationale:** #1 and #2 first — cheapest, highest-EV, most on-theme (per-step
  optimization), both ~segment-safe. #3 is the user-named loss bet. #4/#5 are Tier-2 (low transfer /
  high complexity respectively). Re-rank only with a stated reason in the notebook.
