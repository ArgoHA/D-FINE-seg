# Idea backlog — candidate improvements

Prioritized hypotheses. **Read the paper before implementing.** Bias toward ideas that help
*convergence* (we only train ~60 min → ~22 epochs) and are latency-neutral. Prefer mechanisms that
generalize to COCO; VisDrone-specific and segment-unsafe ones are flagged. Move an idea to
`lab_notebook.md` once tried. Per-idea fields: paper, exact change+files, why it helps under the cap,
latency risk, complexity, expected effect, COCO transfer, segment safety (GUIDE rule 10).

---

## Tier 1 — fast-convergence / latency-neutral

### 1. Scale contrastive denoising (`num_denoising`)  ← top pick: simplest, lowest-risk
- **Paper:** Contrastive DeNoising, DINO (ICLR 2023, arXiv:2203.03605); inherited by RT-DETR/D-FINE.
- **Change (1 key):** `configs.py:19` `num_denoising` `100`→`300`.
- **Why:** groups = `num_denoising // max_gt_num` (`arch/utils.py:380`). Dense VisDrone → large
  `max_gt_num` → floor `num_group=1`: we run the *minimum* denoising. Raising it restores multiple
  noised-GT groups → denser, stable positives exactly when O2O is sparse early.
- **Latency:** none — denoising is `self.training`-gated (`dfine_decoder.py:971`), split off before
  export → graph byte-identical.
- **Complexity:** ~1 line. **Trade-off:** more dn tokens = slightly slower train steps → marginally
  fewer epochs under the cap; log it.
- **Expected:** faster/steadier convergence; modest mAP up, f1 neutral-to-up.
- **COCO:** ✅ general DETR mechanism. **Segment:** ✅ safe/helpful — dn mask loss runs only on the
  final dn layer, normalized by group count (`dfine_criterion.py:752`), so more groups = more mask
  supervision; only extra train cost (mask head over larger dn set).

### 2. DEIM Dense O2O (heavy mosaic)
- **Paper:** DEIM, CVPR 2025 (arXiv:2412.04234). Mosaic/MixUp pack more objects/image → more O2O
  positives/step.
- **Change (1 knob):** `research_visdrone.yaml` `train.mosaic_augs.mosaic_prob` 0.8→1.0 (don't also
  touch the loss).
- **Why:** attacks O2O sparsity, the main convergence bottleneck under the cap.
- **Latency:** none (train-time aug).
- **Complexity:** low. Mind GUIDE §8 (mosaic-close never finishes under cap — consistent, fine).
- **Expected:** convergence speed-up; mAP up if denser supervision beats the harder distribution in
  ~22 epochs.
- **COCO:** ✅ validated on COCO with D-FINE. **Segment:** ⚠️ **harmful — the one real risk here.**
  Mosaic degrades masks (gotcha #6). Keep it a **detect-only** override; never bake high `mosaic_prob`
  into shared/segment defaults. Mechanism does not transfer to segment.
- **MAL note:** MAL (loss half) was tried alone (2026-06-07) and **rejected as a near-tie** — it
  manages the low-quality matches Dense O2O introduces, so it likely only pays *with* Dense O2O.
  After this lands, consider re-testing **MAL + Dense O2O** (two changes; sequence Dense O2O first;
  loss code on `exp/mal`).

### 3. Muon optimizer for 2D matrices (hybrid with AdamW)
- **Paper:** Muon (Jordan et al., 2024) — Newton–Schulz-orthogonalized momentum for 2D weights;
  ~35% faster convergence on nanoGPT speedruns.
- **Change:** in `build_optimizer` (`dfine.py:124`) route encoder/decoder 2D attention/MLP linears
  to Muon, keep backbone/norms/biases/embeddings/**det head**/**mask head** on AdamW. Tune Muon LR
  separately; don't touch the schedule.
- **Why:** encoder/decoder is full of high-condition-number 2D linears (Muon's sweet spot); per-step
  gains matter most under ~22 epochs.
- **Latency:** none (optimizer only; also less optimizer memory).
- **Complexity:** medium (dep + param split + extra LR). Exploratory; drop if marginal (simplicity).
- **Expected:** high-upside, uncertain.
- **COCO:** ✅ architecture-agnostic. **Segment:** ✅ safe **only if** the split excludes `mask_head`
  (a 2D-Linear MLP, `dfine_decoder.py:668`) and `mask_decoder`; stay consistent with the existing
  `enable_mask_head` grouping (`train.py:231`).

### 4. Group-DETR one-to-many auxiliary query groups (training-only)
- **Paper:** Group DETR (arXiv:2207.13085). K query groups, O2O per group, cross-group attention
  masked; inference uses one group → architecture/latency unchanged.
- **Change:** in `dfine_decoder.py` replicate queries into K groups during training, extend the
  self-attn mask (CDN mask at `:971` is a template), match each group O2O in the criterion; drop
  extra groups when `not self.training`.
- **Why:** K× more positives/image, inference unchanged.
- **Latency:** none at inference; heavier train compute/memory.
- **Complexity:** **high** — decoder + matcher. Prefer #1 first (similar benefit, ~1 line).
- **Expected:** convergence speed-up; mAP up.
- **COCO:** ✅ validated on COCO. **Segment:** ⚠️ most segment cost/complexity — mask head runs over
  K× queries and each group needs mask supervision/matching. If built, gate extra groups to
  box/class (skip per-group mask loss) and verify the segment path.

### 5. EMA decay tuned for short runs  ← cheap filler, low transfer
- **Paper:** standard EMA-of-weights; decay sets the averaging window.
- **Change (1 key):** `research_visdrone.yaml` `train.ema_momentum` `0.9998`→`0.999` (leave the
  warmup ramp at `train.py:66`).
- **Why:** `0.9998` ≈ 5000-iter window (tuned for long COCO); in a ~17k-iter still-improving run a
  faster EMA tracks the moving optimum better.
- **Latency:** none. **Complexity:** trivial.
- **Expected:** small mAP/f1 bump.
- **COCO:** ⚠️ **walltime-specific** — long COCO wants high decay back; don't carry the value over.
  **Segment:** ✅ task-agnostic.

---

## Tier 2 — loss / assignment
- Varifocal / quality-focal refinements to the VFL target.
- IoU-aware / distribution-focal regression tweaks (D-FINE already uses FDR/FGL — find orthogonal
  gains, not duplicates).
- Matcher cost reweighting for small/dense objects (`matcher.weight_dict`, `configs.py:42`) — ⚠️
  VisDrone-specific, likely non-transferring; flag for manual COCO check.

## Tier 3 — architecture (watch latency)
- Encoder/neck feature-fusion tweaks; activation/norm swaps; `query_select_method` alternatives
  (`configs.py:25`).

## Notes / constraints
- Init is **ImageNet backbone only** — neck/head start random; backbone starts from HGNetv2.
- Anything adding inference params/FLOPs must clear the *larger* margin AND justify complexity.
- **Segment must not regress (GUIDE rule 10).** Summary: **#2 Dense O2O** is the only directly
  harmful idea (keep detect-only); **#3 Muon** needs the mask head on AdamW; **#4 Group-DETR** is
  costliest for segment; **#1 CDN** and **#5 EMA** are segment-safe. Gate any code/default change on
  `task` if it touches masks.
- **Ranking:** #1 simplest latency-neutral lever and VisDrone is starved (groups→1) → do first; #2
  is the DEIM item that unlocks the MAL re-test; #3/#4 bigger bets with more code; #5 cheap filler.
