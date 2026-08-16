# Lab notebook — D-FINE-seg autoresearch

This file is the **memory across agents/sessions**. A fresh agent reads the *Current state* block
first to know exactly where to resume, then scans entries to avoid repeating dead ends. The
structured numbers live in `ledger.csv`; this file is the reasoning.

## Current state  (keep this block updated every iteration)
- **Baseline established:** yes. ✅ **Horizon-30 re-baseline DONE (2026-06-13, `baseline_h30`)** —
  same current-best Muon recipe re-measured at `train.epochs=30`, 2 seeds. This is now the persisted
  `baseline.json`; the old horizon-100 control (3 seeds, sha `09d0463`) is history. maxDets validator
  fix was **NOT** applied (user decision 2026-06-13: leave detections-per-image as is). Do **not** re-train.
- **Current best (`main_exp`):** 🟢 **Muon + Adan** (Adan promoted 2026-06-14, `baseline.json` →
  name `adan`, sha `4a09ba7`). Enc/dec attn+MLP matrices on Muon (gated by `train.use_muon`); the
  **aux (non-Muon) groups now run Adan** instead of AdamW (gated by `train.aux_optimizer: adan`, aux
  peak LR ×5 via `train.adan_lr_mult`). Backbone/norms/biases/embeds/det+mask heads are the aux groups.
- **Best metrics (test, horizon-30):** mAP_50_95 = **0.2167** , f1 = **0.5635** (seeds .2166/.2169,
  .564/.563; std 0.0002/0.0005→floor margin 0.003 / 0.003; `baseline.json`). Latency-neutral (trt
  2.1ms / torch 13.65ms, ratio 1.0), params 10.302M, TRT row healthy (export OK). **This is the new bar
  for every subsequent candidate** (was Muon-only `baseline_h30` 0.2119/0.5565; Adan added
  +0.0048 mAP / +0.0070 f1, both > margin, multi-seed, zero latency — see 2026-06-14 adan entry).
- 🤖 **Autonomous arch-trio campaign (2026-06-17, user-steered Tier-3 light set) — COMPLETE, 0/3 promoted.**
  A1 SPD-Conv → A3 RMSNorm+SwiGLU → A6 HMC, back-to-back on the S/ImageNet/h30 2-seed screen. Bar = Adan
  0.2167/0.5635 (**HOLDS**). **① A1 SPD-Conv → 🔴** (tie/slight-neg: mAP +0.0006, f1 −0.0020, params
  +0.387M, seed42 TRT-gap −0.004). **② A3 RMSNorm+SwiGLU → 🔴** (positive near-miss: mAP +0.0013, f1
  +0.0010, avg_gain +0.0011 < margin; params +0.788M; **RMSNorm TRT-clean**; RMSNorm-only ablation left
  open). **③ A6 HMC → 🔴** (regression: mAP −0.0101, f1 −0.0235, both past 2× margin; IoU^4 too steep,
  class-cost-×-overlap family dead — extends the PMC tie). Net: the Tier-3 light arch levers don't beat
  the converged design point — consistent with the Tier-3 meta-finding (RT-DETR family stopped touching
  this skeleton). **Methodology call (this campaign):** arch changes alter named params → full `make test`
  pretrained-COCO row fails by construction (no COCO weights for a new graph, §6); gated on `make test-fast`
  (structural/forward/shapes) for A1/A3; A6 (graph-identical) passed full `make test`. Trunk synced
  main_exp→main first (latest frozen harness + torch 2.9.1). **New guide rule added (§5.E):** seed-1
  early-abort when both metrics drop >0.002 (A6's seed42 would have triggered it).
- **In progress:** 🔬 **§6 X full-run A/B** (COCO-init dfine_x_coco, 75ep, batch 4, option-B optimizer
  split: backbone AdamW @4e-6 / heads Adan ×5 @2e-3 / enc-dec Muon @4e-3) vs ref `det_x_2026-02-21`
  (test mAP_50_95 0.2601 / mAP_50 0.4431 / TRT f1 0.611 @4.5ms). **Run-1 = Adan + Muon-WD λ=0.03 DONE
  (2026-06-14)** — user-stopped at best ep46 (overfitting tail, nothing lost): test mAP_50_95 **0.2571
  (−0.0030)**, mAP_50 0.4390 (−0.0041), **TRT f1 0.611 @4.5ms = tie**, torch f1 0.610 @23.2ms. Verdict:
  **not a win** — ties ref on the prod metric, a hair under on mAP; λ=0.03 was marginal/mild-harmful, as
  suspected. Artifacts `experiments/runs/adan_muonwd_x_full75` (+ engine/onnx). **Run-2 = baseline + Adan,
  NO WD DONE (2026-06-15, 9.79h, full 75ep, no early overfit)** — identical recipe minus
  `muon_weight_decay=0.03`. 🟢 **WIN over AdamW-X ref on ALL metrics, latency-neutral:** test mAP_50_95
  **0.2679 (+0.0078)**, mAP_50 0.4543 (+0.0112), **TRT f1 0.617 @4.5ms (+0.006)**, torch f1 0.619.
  Artifacts `experiments/runs/adan_x_full75`. **A/B vs Run-1 is decisive: dropping λ=0.03 WD = +0.0108
  mAP_50_95 swing (0.2571→0.2679), −0.0030 below ref → +0.0078 above.** (Caveat: Run-1 was the one
  stopped at ep46, not a perfectly controlled A/B, but direction unambiguous; WD also caused Run-1's
  early overfit.) **Run-3 = "just Muon" DONE (2026-06-15, stopped ep71, overfitting; best kept)** on the
  FIXED code with the flag REMOVED → validates the l/x size-swap end-to-end (ran correctly, no NaN, sane
  win). test mAP_50_95 **0.2626 (+0.0025)**, mAP_50 0.4466 (+0.0035), **TRT f1 0.613 @4.5ms (+0.002)**,
  torch f1 0.616. Artifacts `experiments/runs/muon_x_full75`. (`run_muon_x_full75.sh`, `aux_optimizer=adamw`
  + Muon; only delta vs ref is enc/dec→Muon.)
- 🟢 **A/B/C VERDICT at X (all latency-neutral, TRT 4.5ms = ref):** clean decomposition — **+Muon alone
  = +0.0025** mAP_50_95 (small real lift); **+Adan on top = +0.0053** (Run-2 0.2679 − Run-3 0.2626,
  ~2× Muon's share); **−WD λ=0.03 = harmful** (rejected). **Muon+Adan (Run-2, +0.0078) is the best X
  recipe.** Key finding: **Adan scales** — it added +0.0048 over Muon at S (horizon-30) and +0.0053 at X
  (full-75); near-identical, killing the "Adan is an S-only artifact" hypothesis. (Horizon caveat:
  Run-2 full-75, Run-3 ep71, Run-1 ep46 — each at its overfit point, best kept, near-final but not matched.)
  **DECISION (user, 2026-06-15):** (1) l/x LR fix **PROMOTED to main_exp** (commit `e2e5ef9`, code only;
  `muon_weight_decay` deliberately NOT included — it's the rejected muon-wd-003 knob); (2) X win
  **ACCEPTED as-is** (single-seed full runs, no further X confirmation) → **resume the S autoresearch loop.**
- 🔧 **Muon l/x LR fix LANDED (this branch, 2026-06-14):** `train.py` now auto-forces
  `respect_backbone_lr` True for l/x. **Root cause:** the Muon scheduler branch (train.py ~247) *overwrote*
  the per-group `max_lr` list the non-Muon path already builds for l/x (`backbone_lr*2` for backbone
  groups), homogenizing the B5 backbone to `base_lr*2` (×5 for Adan) → up to **500× too high**, frying
  the pretrained backbone. Never bit because the loop only runs S (single scalar max_lr there; tiny B0
  tolerates it). Fix is S-byte-identical (`s` ∉ l/x). Verified X auto-default `max_lr=[4e-6,4e-6,4e-4,
  4e-4,4e-3]` with the flag REMOVED → run-3 validates `model_name=x` size-swap. **PROMOTED to main_exp
  2026-06-15 (commit `e2e5ef9`: dfine.py+train.py+config.yaml, S byte-identical, 78 unit tests pass).**
  The cleaner follow-up (also cover `enable_mask_head`, drop the flag entirely) is open.
  🔬 **QK-norm arc CLOSED → SHELVED, knowledge preserved (2026-06-10, see `experiments/qk_norm.md`).** QK-norm solves the issue-#64 NaN class and at full 75-ep scale even **beats muon_full75 on training metrics (test mAP_50_95 0.2388 vs 0.2359, +0.0029)** — but TensorRT **mis-executes the fully-trained checkpoint at ALL precisions** (fp32 0.552 / fp16 0.545 vs torch/ORT-true 0.582; TRT 10.16/11.0 strictly worse). It's a *weights-dependent* TRT compiler defect: a structurally identical ONNX from the screen checkpoint compiles correctly; every op is fine in isolation; ORT/torch always agree. Since the deliverable is the fp16 TRT engine → code stays OFF the trunk (full impl + revisit conditions in `qk_norm.md`; branch `exp/qk-norm-lr`). **Muon stays best.** Two durable wins landed on the way: (1) **TRT fp16 export hardening** in `dfine_seg/dl/export.py` — strong-typed engine with GridSample pinned fp32; without it full-fp16 silently costs even the muon model −0.026 f1 (0.585→0.559), with it muon is at exact parity 0.585 @ 2.1 ms (TRT 11 removes auto-FP16, so this is also the forward-compatible path); (2) the f1 guard reads the **TensorRT** bench row (guide §3 + `run_candidate.py`), which is exactly what caught all of this.
  ✅ **Full 75-epoch Muon confirmation DONE (2026-06-08)** — COCO-init, 75ep,
  no cap, single seed (`experiments/runs/muon_full75/seed42`). **test mAP_50_95 0.2359 vs ref 0.2316
  (+0.0043), val 0.2965 vs 0.2882 (+0.0083), mAP_50 0.4063 vs 0.3995, f1 0.5633 vs 0.5621, latency
  neutral (trt 2.1 / torch 13.3 ms).** Fair comparison (Muon = non-arch, COCO weights load identically).
  Key: the +0.0043 test gain is **identical to the 22-epoch proxy gain** → Muon reaches a *better
  optimum*, not just faster convergence (an AdamW catch-up would have shrunk the gap by ep75). Single
  seed, but proxy(+0.0043, clean same-code) and full(+0.0043 vs Feb ref) agreeing rules out seed/code-drift luck.
- **User-approved 3-experiment Tier-2 train-only set (2026-06-14): precise-bn → adan → muon-wd λ=0.03.**
  ① **#9 PreciseBN → 🔴 tie/no-op** (guard reverted both seeds; BN-gap falsified). ② **#11 Adan → 🟢
  PROMOTED** (mAP 0.2167, +0.0048; f1 0.5635, +0.0070; both > margin, multi-seed std 0.0002/0.0005,
  zero latency, no NaN — new best). ③ **Muon-WD λ=0.03 — IN PROGRESS next**, now tested **on top of the
  new Adan baseline** (Muon-group WD is orthogonal to the aux Adan change, so the follow-up is still
  clean; bar is now 0.2167/0.5635).
- **Next idea: Adan promoted → optimizer axis is alive again.** TIER-1 EXHAUSTED (5/5 🔴); Tier-2 #10
  backbone-LR 🔴 tie, #9 PreciseBN 🔴 tie/no-op; **#11 Adan 🟢 PROMOTED (new best 0.2167/0.5635).**
  Running #3 of the approved set next: **Muon-WD λ=0.03** (deferred #4 follow-up — λ=0.1 over-regularized
  the short screen, λ=0.03 τ≈6.7k is gentler; vs the Adan bar now). After that: #6 EMA bracket, and the
  **§6 full-run / COCO confirmation of Adan** (non-arch change → COCO-init is a fair bar, like Muon got;
  manual, user-triggered). Mechanistic read updated: **the optimizer axis is where the signal lives —
  Muon (enc/dec matrices) AND now Adan (aux groups) both moved the screen.** Matcher-cost (PMC),
  optimizer-update-shaping (Cautious/Moonlight/Muon-WD λ=0.1), cls-target (MAL/IA-BCE), LR-ratio
  (backbone-LR), and BN-stats (PreciseBN) were all probed and did not beat the bar.
  ideas.md was fully rewritten 2026-06-13 after a deep-research pass (5 Tier-1 +
  7 Tier-2, all train-only); the old MAL-on-Muon re-test is **withdrawn** (DEIM never ablates MAL
  standalone — our tie matches the paper). QK-norm remains shelved (TRT-undeployable; recipe in
  `experiments/qk_norm.md`); if real-user stability ever bites (issue #64), QK-norm is the known
  torch-side fix and the robustness net (clamp `pred_corners` + NaN-safe GIoU) is the orthogonal
  `main` hardening. Standing conclusion unchanged: the cap bottleneck is per-step optimization
  quality, not positive count (CDN, Dense O2O rejected; Muon landed).
- **Notes for the next agent:**
  - **Methodology change (2026-06-13): schedule horizon `train.epochs` 100 → 30**, plus mosaic
    close pinned off (`train.mosaic_augs.no_mosaic_epochs: 0`) for cross-seed determinism. 60 min ≈
    21 epochs → runs now end ~65-80% through the anneal (~8-30% of peak LR) instead of at ~96% of
    peak, so screen verdicts are measured near a converged run's end state (user decision; the
    cooldown/aug-close *candidate* ideas were rejected as screen-regime-only — improvements must
    show in the standard setup). **Horizon-30 re-baseline DONE 2026-06-13** (`baseline_h30`: test
    mAP 0.2119 / f1 0.5565) — the old 0.2061/0.552 bar was horizon-100 history. Same day: guide §0
    mission rewritten (improve D-FINE-seg generally; VisDrone is only the screen), ideas.md fully
    rewritten with sources. The validator maxDets=100 under-measurement was flagged for user
    sign-off (ideas.md §Methodology) but **deliberately NOT changed** — user decision 2026-06-13 to
    leave detections-per-image as is; validator.py stays frozen/unmodified.
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

## 2026-06-17 — hmc (Rank-DETR high-order matching cost, Tier-3 A6)   [rejected — regression]
- Paper / source: Rank-DETR (NeurIPS'23, arXiv:2310.08854) high-order matching cost. ideas.md Tier-3 A6.
  Third/final of the autonomous arch trio (A1→A3→A6), 2026-06-17.
- Hypothesis: weight the Hungarian class cost by IoU^4 so matching favors jointly high-confidence +
  well-localized queries (steeper than the gentle PMC ((GIoU+1)/2)^0.5 that tied earlier). Train-only
  (matcher @no_grad), graph-identical → zero TRT/latency risk.
- Change (files): `matcher.py` — import box_iou; after cost_giou, `cost_class = cost_class *
  iou.clamp(min=0).pow(4)`. ~4 LOC. exp/hmc sha bc8bbe8. **Full `make test` 89/89** (graph-identical →
  pretrained-COCO row passes, unlike the A1/A3 arch changes).
- Result (test, 2 seeds): mAP_50_95 **0.2066±0.0019** (seeds .2047/.2085, gain **−0.0101**, past 2×
  margin ❌), f1 **0.54±0.002** (TRT row, seeds .538/.542, gain **−0.0235** ❌), lat trt 2.1 / torch
  13.85 ms (ratio 1.0), params 10.302M (unchanged — train-only). trt_export_flagged []. No NaN. 🔴 KEEP BEST.
- Read: clear **regression on both metrics**, the worst of the trio. IoU^4 is **far too steep** for this
  matcher: it zeroes the class cost for every pair below near-perfect IoU, so early in training (when
  predicted IoU is low across the board) the matcher loses its classification signal and assigns almost
  purely on bbox/giou — exactly the churn PMC was meant to *reduce*, here amplified. Confirms and extends
  the PMC tie (#1): the class-cost-×-overlap family is dead here, and steeper is strictly worse (PMC ^0.5
  tied, HMC ^4 regresses −0.01/−0.024). A milder exponent (α=1-2) might recover toward the PMC tie but not
  beat it → not worth a slot. **Validates the new seed-1 early-abort rule (§5.E, added this session):**
  seed42 alone was −0.0120 mAP / −0.0255 f1 (both ≫ 0.002 below baseline) → the rule would have killed
  after seed42 and saved the seed123 hour. Segment: shared matcher; rejected → n/a. **Autonomous arch trio
  COMPLETE: A1 🔴 / A3 🔴 / A6 🔴 — none promoted; Adan (0.2167/0.5635) holds.**

## 2026-06-17 — rmsnorm-swiglu (DEIMv2 decoder modernization, Tier-3 A3)   [rejected — positive near-miss]
- Paper / source: DEIMv2 "Real-Time Object Detection Meets DINOv3" (arXiv:2509.20787) efficient decoder.
  ideas.md Tier-3 A3. Second of the autonomous arch trio (A1→A3→A6), 2026-06-17.
- Hypothesis: modernize the decoder layer the way DEIMv2 does — LayerNorm→RMSNorm (drops mean-subtraction,
  cheaper + an fp16-stability win) and the ReLU-MLP FFN→SwiGLU (gated, strictly more expressive).
  Deformable cross-attn + FDR + LQE + CDN kept verbatim. linear1/linear2 names preserved so Muon still
  captures them (no optimizer-group confound).
- Change (files): `arch/dfine_decoder.py` TransformerDecoderLayer — norm1/norm3 nn.LayerNorm→nn.RMSNorm;
  FFN→SwiGLU (linear1 d_model→2*dim_feedforward, chunk gate/value, F.silu(gate)*value, linear2 unchanged).
  Gate's internal LayerNorm + fp16 clamp left as-is; shared decoder pos-embed deliberately NOT hoisted
  (fights the FDR cascade). ~6 LOC. exp/rmsnorm-swiglu sha 8dc49aa. `make test-fast` 87/87 (full make test
  pretrained-COCO row N/A — arch change).
- Result (test, 2 seeds): mAP_50_95 **0.218±0.0007** (seeds .2187/.2173, gain **+0.0013**), f1
  **0.5645±0.0005** (TRT row, seeds .565/.564, gain **+0.0010**), avg_gain **+0.0011 < margin 0.003**,
  lat trt 2.1 / torch 13.55 ms (ratio 1.0), params **11.09M (+0.788M)**. trt_export_flagged [] — **clean
  export, both seeds gap −0.002** (RMSNorm→ONNX→TRT fine; no qk-norm-class footgun). No NaN. 🔴 KEEP BEST.
- Read: **positive near-miss** — both metrics up, latency-neutral, export clean, but avg_gain (+0.0011) is
  ~⅓ of the 0.003 margin. The decoder modernization genuinely helps a hair (consistent with DEIMv2 adopting
  it wholesale), just not past screen noise — and it costs **+0.788M params** (SwiGLU's doubled linear1), so
  the simplicity rule (1.6) seals the keep: a sub-margin gain doesn't justify the arch complexity + param
  growth. Notable positive: **RMSNorm is TRT-clean** (the feared LayerNorm→RMSNorm export issue did not
  materialize) → a safe stability building block if issue-#64-class NaNs ever bite. Open follow-ups (each a
  separate experiment, not run): (a) **ablate which half drove +0.0013** — RMSNorm-only (zero param cost)
  vs SwiGLU-only; if RMSNorm-only keeps most of the gain it'd be a free, simpler, promotable change worth a
  slot; (b) param-matched SwiGLU (shrink dim_feedforward ×⅔) to drop the param penalty. Segment: shared
  decoder; rejected → n/a here. Next: A6 HMC (train-only matcher filler, trio ③).

## 2026-06-17 — spd-conv (SPD-Conv neck PAN downsample, Tier-3 A1)   [rejected — tie / slight-neg]
- Paper / source: SPD-Conv (Sunkara & Luo, arXiv:2208.03641, ECML-PKDD'22). ideas.md Tier-3 A1. First
  of the user-approved autonomous arch trio (A1 SPD-Conv → A3 RMSNorm+SwiGLU → A6 HMC), 2026-06-17.
- Hypothesis: strided downsampling discards the high-freq detail tiny objects live on (55% of our boxes
  <16px). Replace the PAN bottom-up SCDown (1x1 + depthwise stride-2) with space-to-depth (slice into 4
  stride-2 sub-maps, concat → 4C at H/2×W/2) + a 1x1 conv to restore channels — moves detail into
  channels instead of dropping it. TRT-safe (Slice+Concat, no grid_sample); fair (neck change, backbone
  ImageNet weights untouched; the 2 downsample convs init random under strict=False — partial-init).
- Change (files): `arch/hybrid_encoder.py` — new `SPDConv` (space-to-depth + ConvNormLayer_fuse(4C,C,1,1));
  swapped into `downsample_convs` (was `SCDown(hidden_dim,hidden_dim,3,2)`). ~10 LOC. exp/spd-conv sha
  3be4729. `make test-fast` 87/87 (full `make test` pretrained-COCO row fails by construction — new graph,
  no COCO weights; forward healthy 14TP/0FP/5FN, mAP_50_95 0.712≥0.7).
- Result (test, 2 seeds): mAP_50_95 **0.2173±0.0007** (seeds .2166/.218, gain **+0.0006** ≪ margin),
  f1 **0.5615±0.0005** (TRT row, seeds .561/.562, gain **−0.0020**, within margin), avg_gain −0.0007,
  lat trt 2.1 / torch 13.25 ms (ratio 1.0), params **10.689M (+0.387M)**. trt_export_flagged [seed42]:
  torch f1 0.565 vs TRT 0.561 (gap −0.004, just over the 0.003 warn tol; not a collapse — both healthy).
  No NaN. 🔴 KEEP BEST.
- Read: clean **tie / slight-negative** — the first DETR-family SPD-Conv test does not transfer the YOLO
  small-object win here. Likely reasons: (1) the swap is at the **neck PAN** (stride 16/32), not the early
  high-res backbone where SPD's detail-preservation pays most — ideas.md's higher-upside placement (b),
  the backbone stem, was deferred as the bigger arch change; (2) D-FINE's RepNCSPELAN4 neck + deformable
  decoder already aggregate multi-scale context, so a detail-preserving downsample at 1/16–1/32 adds
  little. The +0.387M params + the seed42 TRT-gap flag (a faint Slice+Concat fragility signal) make this
  **not** worth keeping even at a tie (simplicity rule). Rejected → code stays on exp/spd-conv for
  forensics; the backbone-stem placement (b) remains an open, larger arch bet if SPD is revisited.
  Segment: feeds the mask-head stride-8 tap region but masks untouched on this detect screen (rejected →
  n/a). Next: A3 RMSNorm+SwiGLU decoder modernization.

## 2026-06-14 — adan (Adan optimizer on the aux/non-Muon groups, Tier-2 #11)   [PROMOTED — second real win]
- Paper / source: Adan (Xie et al., arXiv:2208.06677, TPAMI'24) — adaptive Nesterov momentum; the only
  modern optimizer with a published DETR-family COCO win (Deformable-DETR-R50 50e 44.5→45.3, +0.8 over
  tuned AdamW; Mask R-CNN +0.5 box/+0.5 mask). ideas.md Tier-2 #11. Experiment 2 of the user-approved set.
- Hypothesis: the optimizer axis is the only lever that has moved this screen (Muon). Muon already owns
  the enc/dec matrices; the **aux groups** (backbone + det/mask heads + norms/biases) still run vanilla
  AdamW. Swap them to Adan — its gradient-difference (Nesterov) term + per-coord second moment should give
  better per-step progress on exactly those params. Train-only, zero latency/TRT risk.
- Change (files): vendored single-tensor `_adan_update` in `dfine_seg/model/muon.py` (3 buffers m/v/n +
  `neg_pre_grad`, betas (0.98,0.92,0.99), eps 1e-8, decoupled post-prox WD — faithful to the sail-sg
  reference, minus global-grad-clip/restart); branched into `MuonWithAuxAdam.step()` for aux groups via a
  per-group `aux_optimizer` flag. `dfine.py:build_optimizer` gains `aux_optimizer="adamw"` → flags the 4
  aux groups + sets Adan betas/eps (per-group WD kept identical to the AdamW recipe, so only the update
  rule + LR change). `train.py` threads `train.aux_optimizer` and scales the aux OneCycle peak LR ×5
  (`train.adan_lr_mult`, Adan's convention) — Muon group's `muon_lr*2` peak untouched. `config.yaml`
  defaults `aux_optimizer: adamw` / `adan_lr_mult: 5.0` (Hydra struct + prod off). Detect-screen override
  `train.aux_optimizer: adan` in `research_visdrone.yaml` (on-disk, git-ignored). exp/adan sha `4a09ba7`.
  `make test` 89/89; numerically smoke-tested (Adan finite, ≠ AdamW on same grads, first-step diff=0,
  neg_pre_grad stores −grad); verified group structure on the real model (4 aux groups Adan, 25-matrix
  Muon group untouched, aux peak 0.0025 / muon peak 0.005).
- Result (test, 2 seeds): mAP_50_95 **0.2167±0.0002** (seeds .2166/.2169, gain **+0.0048** > margin
  0.003), f1 **0.5635±0.0005** (TRT row, seeds .564/.563, gain **+0.0070** > margin), lat trt 2.1 /
  torch 13.65 ms (ratio 1.0), params 10.302M. **No NaN events** (the feared ×5-aux-LR backbone NaN did
  not bite — Adan's normalized update keeps the effective step ~lr-bounded). TRT row healthy → no export
  regression. 🟢 PROMOTE.
- Read: **second clean win after Muon, and bigger** (Muon was +0.0043 mAP; Adan +0.0048). Both metrics
  clear the margin with extremely tight seeds (std 0.0002/0.0005), latency flat, no NaN, and the f1 GUARD
  *improved* (+0.0070) — a strict dominance, not a trade. Confirms the standing diagnosis sharpened:
  **the live lever is per-step optimization quality, and it has now paid off on BOTH parameter blocks** —
  Muon on the high-condition enc/dec matrices, Adan on the aux groups (backbone/heads/norms). Adan's win
  is consistent with its published DETR result, so it should transfer (the §6 COCO-init full-run is a fair
  confirmation since this is a non-arch change — deferred/manual, like Muon's). **Simplicity check
  (rule 1.6):** ~45 lines, default-off, isolated inside the existing optimizer module — comparable to
  Muon's ~90 lines, for a larger and cleaner win → complexity justified, promote. **Confound note:** Adan
  ships with a ×5 aux-LR (its convention); this is "Adan + its recommended LR" as one conceptual unit
  (exactly how Muon = optimizer + its own LR was judged), not two free variables — the AdamW LR would
  mis-tune Adan. WD was deliberately *not* changed (kept the repo's per-group values) to isolate the
  optimizer. **Segment safety:** Adan now drives the **mask-head** optimizer (mask params are in the aux
  groups). Adan is seg-validated in its own paper (Mask R-CNN +0.5 mask), but verify masks don't regress
  before/at any segment release. Follow-ups: (a) §6 Adan COCO full-run; (b) Muon-WD λ=0.03 (experiment 3,
  now on the Adan baseline); (c) Adan LR-mult sweep (×3/×8) is a *retune*, low priority after a clean win.

## 2026-06-14 — precise-bn (PreciseBN BN-stat recalibration, Tier-2 #9)   [rejected — tie / no-op]
- Paper / source: Wu & Johnson, "Rethinking 'Batch' in BatchNorm" (arXiv:2105.07576). ideas.md Tier-2 #9.
  First of the user-approved 3-experiment Tier-2 train-only set (precise-bn → adan → muon-wd λ=0.03).
- Hypothesis: the campaign trains on 80% mosaic and **never closes it** (pinned `no_mosaic_epochs:0`), so
  the EMA BN running stats are estimated on a collage distribution the clean eval never sees. Recompute BN
  population stats on clean (val-transform) train data after the cap → fix the train/eval BN mismatch at
  ~free cost (runs outside the 60-min walltime).
- Change (files): new `train.precise_bn` / `train.precise_bn_batches` knobs (`config.yaml` defaults
  False/200 — Hydra struct-mode needs them declared); module-level `update_precise_bn` in `train.py`
  (model stays `eval()` → inference path, no CDN/targets; only `_BatchNorm` modules switch to
  cumulative-average tracking via `reset_running_stats()` + `momentum=None`); wired into the `finally`
  block with a **keep-if-better guard** — eval val mAP_50_95 before/after, overwrite `model.pt` only if
  PreciseBN ≥ EMA else revert (so the change can never regress, and export/bench use the winner). Clean
  (val-transform) train-images loader built from `base_loader` (CustomDataset over train split, mode=val).
  Detect-only override `train.precise_bn: true` in `research_visdrone.yaml` (on-disk; that file is
  git-ignored). exp/precise-bn sha `cf4d8fc`. `make test` 89/89; Hydra override verified.
- Result (test, 2 seeds): mAP_50_95 **0.2117±0.0013** (seeds .2129/.2104, gain **−0.0002**), f1
  **0.554±0.001** (TRT row, seeds .555/.553, gain −0.0025, within margin), lat trt 2.1 / torch 13.6 ms
  (ratio 1.0), params 10.302M. **PreciseBN REVERTED on BOTH seeds** (guard fired): val mAP_50_95
  0.2602→0.2571 (s42) and 0.2607→0.2565 (s123) — recomputing BN stats made val *worse*. 🔴 KEEP BEST.
- Read: clean **tie / no-op**. Because the keep-if-better guard reverted both seeds, the candidate
  `model.pt` is byte-equivalent to the baseline recipe (no BN change applied) → the result IS the baseline
  re-run, and the tiny deltas are pure seed/walltime-stop variance (baseline was .2114/.2124, f1 .556/.557).
  The substantive finding is the **revert itself**: the "mosaic→clean BN distribution gap" hypothesis does
  **not** hold for this model. Likely why — (1) HGNetv2-B0 has relatively few BN layers and the EMA running
  stats (τ≈5k steps) already average over a long, well-mixed window that tracks the eval distribution fine;
  (2) recomputing over a 200-batch (1600-img) sample is a *higher-variance* estimate than the long EMA
  average, so it adds noise rather than removing bias. The guard worked exactly as designed — zero
  regression risk, and the negative is informative (BN-stat staleness is not a lever here). Code stays
  default-off; rejected → nothing lands on trunk. Segment: ✅ (BN-stat recompute is task-agnostic, mask head
  untouched). **Next: #11 Adan** (experiment 2 of 3).

## 2026-06-13 — backbone-lr (backbone-LR ratio raise, Tier-2 #10)   [rejected — tie]
- Paper / source: RT-DETRv2 (arXiv:2407.17140) scales backbone LR by capacity — its lightest backbone
  (R18) runs at ratio 1.0 to the head LR. ideas.md Tier-2 #10. First Tier-2 item (Tier-1 exhausted).
- Hypothesis: ours runs HGNetv2-B0 at ratio 0.24 (backbone_lr 6e-5 / base_lr 2.5e-4) — a heavy-backbone
  value — under ImageNet-only init + a large VisDrone domain gap, so the backbone is plausibly
  under-trained. Raise 6e-5 → 1.2e-4 (ratio ~0.5). A win also lands as a better user-facing per-size
  default in the LR table. Config-only (1 key), zero code/TRT/segment risk.
- Change (files): **config-only** — `research_visdrone.yaml` override `train.lrs.s.backbone_lr:
  0.00012`. No code change → no commit on exp/backbone-lr (candidate = on-disk config over main_exp
  `b3c0228`). `make test` 89/89; verified override resolves (backbone_lr 0.00012 / base_lr 0.00025).
- Result (test, 2 seeds, tight): mAP_50_95 **0.2118±0.0009** (seeds .2109/.2128, gain **−0.0001**),
  f1 **0.5575±0.0015** (TRT row, gain +0.0010), lat trt 2.1 / torch 13.6 ms (ratio 1.0). **No NaN
  events** (the feared backbone-LR NaN amplification did not bite at 2×). 🔴 KEEP BEST.
- Read: clean **tie** — neutral, not a regression. The cold-backbone hypothesis doesn't pay off at the
  horizon-30 / ~21-epoch screen: under the walltime cap the backbone gets few enough updates that
  doubling its LR neither meaningfully speeds adaptation nor destabilizes it (f1 nudged +0.0010, mAP
  flat). The documented follow-up (ratio ~0.8 → backbone_lr 0.0002) is **not pursued** — a tie at 0.5
  gives it low prior, and the mechanism is LR-tuning (no new capability). Notably this is the first
  non-regressing non-Muon result in a while, reinforcing that LR/optimizer is the only live axis but
  that Muon already captures the reachable gain there. Segment: ✅ (LR config only); rejected → no trunk
  change regardless.

## 2026-06-13 — ia-bce (IoU-aware classification target, Align-DETR IA-BCE)   [rejected — regression]
- Paper / source: Align-DETR (arXiv:2304.07527, BMVC'24) IA-BCE — +1.3 vs VFL head-to-head (DINO-R50
  12e: 48.7→50.0), AP_S +2.7–3.7. ideas.md Tier-1 #5 (the IoU-aware cls slot; one of IA-BCE/GCL).
- Pre-step (free, no train run): a matched-IoU diagnostic (script since removed) — ran the **baseline_h30** ckpt +
  the training matcher over the val set, histogrammed matched-pair IoU (the `ious` loss_labels_vfl
  uses). Result: mean 0.66, **median 0.73, only 6.2% < 0.1**, 79.5% ≥ 0.5. Low IoU≈0 mass → the GCL
  trigger does NOT fire → **pick IA-BCE** (overturned the size-distribution prior of GCL: 55% sub-16px
  GT, but tiny objects are either localized well or *missed* — FNs aren't matched pairs, so they don't
  add IoU≈0 mass). Useful lesson: matched-IoU ≠ GT-size distribution.
- Hypothesis: replace VFL's IoU target with IA-BCE — pos soft target `t = s^0.25·u^0.75` (s = pred
  score at matched gt class), pos weight 1, neg focal weight `s^2`. Better aligns cls score with
  localization quality; the published direct VFL replacement.
- Change (files): `dfine_criterion.py:loss_labels_vfl` IA-BCE branch + `cls_loss` flag on the
  criterion; `build_loss`/`train.py` thread it; `config.yaml` default `cls_loss: vfl`
  (prod/segment unchanged); `research_visdrone.yaml` set `ia_bce` (**detect-only** — shared cls loss).
  exp/ia-bce sha `cf319f7`. `make test` 89/89; smoke-tested IA-BCE finite & ≠ VFL.
- Result (test, 2 seeds): mAP_50_95 **0.2098±0.0013** (seeds .2111/.2085, gain **−0.0021**), f1
  **0.5405±0.0015** (TRT row, gain **−0.016**, **guard ❌ regressed hard**), lat trt 2.1 / torch 13.55
  ms (ratio 1.0). No NaN. 🔴 KEEP BEST.
- Read: clear **regression**, worst on f1. The big f1 drop persists *even at the val-optimal threshold*
  (so it's not just the expected score-operating-point shift — it's genuine degradation at the best
  operating point). Likely cause: IA-BCE's negatives carry weight `s^2` with **no α (0.2) down-weight**
  (vs VFL), so the cls loss runs ~4–5× hotter relative to the unchanged bbox/giou/local terms — the
  loss balance shifts against localization without a `weight_dict` re-tune, which on D-FINE's
  fine-grained-regression recipe hurts precision/recall. The 12-e COCO +1.3 was measured on a
  VFL-baseline DINO with its own loss weights; it doesn't transfer to D-FINE's balance here. A
  `loss_vfl` weight re-tune *might* rescue it, but that's a second variable (one-change rule) and
  low-prior — **not pursued**. GCL untested (pre-step pointed to IA-BCE); could be a future slot but
  the whole cls-target family (MAL tie, IA-BCE regress) is now low-prior. Segment: detect-only override
  → segment stays on vfl, **no segment impact**; rejected → nothing on trunk regardless.

## 2026-06-13 — muon-wd (real decoupled weight decay on the Muon group)   [rejected — regression]
- Paper / source: Moonlight (arXiv:2502.16982) Fig.2 (vanilla Muon grows weights, ends worse; λ=0.1
  wins); Kimi K2 (2507.20534) attn-logit explosions; timescale rule (2405.13698) τ_wd=1/(η·λ). Tier-1 #4.
- Hypothesis: the global WD (1.25e-4) is **inert** on this screen — τ_wd≈1.6e7 steps vs an ~18k-step
  run. Put real decoupled λ=0.1 on the Muon group only (τ≈2k steps at peak Muon LR, Moonlight's
  operating point) to bound weight/attn-logit growth → quality + stability. WD's benefit is supposed
  to *grow* with run length, so a truncated screen under-measures it (asymmetry flagged in ideas.md).
- Change (files): new `train.muon_weight_decay` knob (`config.yaml` default null → global WD);
  `build_optimizer` routes it onto the Muon group only (verified: Muon group wd=0.1, all AdamW groups
  unchanged at 1.25e-4/0.0, legacy muon_lr=base_lr×10 preserved). `research_visdrone.yaml` set 0.1.
  exp/muon-wd sha `47550f3`. `make test` 89/89.
- Result (test, 2 seeds, tight): mAP_50_95 **0.2057±0.0008** (seeds .205/.2065, gain **−0.0062**, 2×
  margin), f1 **0.55±0.002** (TRT row, gain −0.0065, **guard ❌ regressed**), lat trt 2.1 / torch 13.6
  ms (ratio 1.0). No NaN events. 🔴 KEEP BEST.
- Read: clear **regression**, larger than Moonlight's — λ=0.1 **over-regularizes** on the ~18k-step
  horizon. τ≈2k steps means the Muon-group weights are decayed ~9× over the run; on a short schedule
  that shrinks useful capacity faster than convergence can use it (Moonlight's λ=0.1 is tuned for
  100k+-step LLM runs). The mechanism (inert global WD → real WD on Muon) is sound; the *level* is
  wrong for this length. Two documented, **deferred** follow-ups (not run — only #5 was approved next):
  (1) **λ=0.03 down-check** (τ≈6.7k steps — gentler, ideas.md's own fallback); (2) the **§6 full-run**,
  where WD's benefit grows with length and the screen is expected to under-measure it — a flat/negative
  screen is the *expected* signal here, so this is the one rejected idea whose full-run check is
  genuinely motivated before final discard. For now the inert global WD stays (prod unchanged); knob
  kept (null→legacy). Segment: Muon group only, mask head stays AdamW; rejected → nothing on trunk.

## 2026-06-13 — moonlight-rms (Moonlight update-RMS match for the Muon group)   [rejected — regression]
- Paper / source: Moonlight / "Muon is Scalable for LLM Training" (arXiv:2502.16982) Eq.4. ideas.md Tier-1 #3.
- Hypothesis: rescale the orthogonalized Muon update by `0.2·sqrt(max(A,B))` so its update-RMS
  matches AdamW's (~0.2) per matrix shape, and **re-anchor the Muon-group peak LR to base_lr** (was
  base_lr×10). Today's fan-shape scaling runs the Muon group ~3× hotter than AdamW-RMS parity; the
  rescale is cooler overall + a per-shape reallocation (square attn ×3.2, wide FFN down-proj ×6.4) a
  global multiplier can't express. Closes the "muon_lr is a blind ×10" open question either way.
- Change (files): `dfine_seg/model/muon.py:38` scaling line → `0.2*max(A,B)**0.5`; new `train.muon_lr`
  knob (`config.yaml` default null → legacy base_lr×10) read in `train.py`; `research_visdrone.yaml`
  set `train.muon_lr: ${train.base_lr}` (= 0.00025 for s, vs legacy 0.0025). exp/moonlight-rms sha
  `4e68f07`. `make test` 89/89; verified the interpolation resolves to a numeric LR (not a string).
- Result (test, 2 seeds, tight): mAP_50_95 **0.2091±0.0005** (seeds .2086/.2096, gain **−0.0028**),
  f1 **0.554±0.0** (TRT row, gain −0.0025), lat trt 2.1 / torch 13.65 ms (ratio 1.0), params 10.302M.
  No NaN events (the cooler setting is stable). 🔴 KEEP BEST.
- Read: clear **regression** on both metrics (not a tie) — the principled RMS-parity setting is *worse*
  here than the accidental 3×-hot legacy scaling. This **answers the open "blind ×10" question**: on
  this horizon-30 screen the hotter Muon group is genuinely better, not just lucky — the enc/dec
  matrices tolerate (want) the higher effective step, consistent with why the global raise to 0.01
  NaN'd (too hot *globally*) yet base_lr×10 *on the Muon group only* is the sweet spot. Net: legacy
  scaling vindicated and now documented; the `muon_lr` knob stays in (null→legacy, prod unchanged) as
  a useful exposed default for future LR work, but the trunk keeps the ×10. Follow-up "too cold"
  knob (muon_lr = base_lr×1.5–2, still shape-correct) is **not** worth a slot — the *shape rescale
  itself* is the regressor here, not just the LR level (LR re-anchor and rescale moved together, but
  the result is decisively worse, so re-warming alone is unlikely to recover). Segment: Muon-group
  only, mask head stays AdamW; rejected → nothing on trunk, no segment impact.

## 2026-06-13 — cautious (Cautious AdamW on the aux/AdamW groups)   [rejected — tie]
- Paper / source: "Cautious Optimizers" (arXiv:2411.16085, NeurIPS'24); timm replication
  (rwightman/timm-optim-caution: vit_wee mini-IN 71.23→73.52). ideas.md Tier-1 #2.
- Hypothesis: zero AdamW-group update coords whose sign disagrees with the live grad, renorm by
  mask density ("don't step where unsure") → strictly better per-step progress on the half of params
  Muon doesn't touch (backbone + det head + norms/biases/embeds). Train-only, zero latency/TRT risk.
- Change (files): `dfine_seg/model/muon.py` AdamW branch — `m=(upd*p.grad>0); m/=m.mean().clamp(1e-3);
  upd*=m`, gated by `cautious` flag; threaded via `dfine.py:build_optimizer` + `train.py`;
  `config.yaml` default `train.cautious: False` (Hydra struct-mode needs the key declared); enabled
  via `train.cautious: true` in `research_visdrone.yaml`. exp/cautious sha `ea1bffd`. `make test` 89/89.
  (First launch died instantly — Hydra rejected the undeclared `train.cautious` override; fixed by
  adding the `config.yaml` default, then relaunched.)
- Result (test, 2 seeds): mAP_50_95 **0.2134±0.0018** (seeds .2116/.2151, gain **+0.0015**, < 0.003
  margin), f1 **0.5585±0.0015** (TRT row, seeds .557/.560, gain +0.0020, within margin), lat trt 2.1 /
  torch 13.45 ms (ratio 1.0), params 10.302M. 🔴 KEEP BEST. (TRT bench row healthy — no export regression.)
- Read: clean **tie** — both metrics nudge up but neither clears the margin. The mAP seed spread
  (0.0035) is marginally > the 0.003 margin (rule 9 flags a possible 3rd seed), but the verdict is
  robust: to flip reject→promote the mean must clear +0.003 (>0.2149), needing a 3rd seed >0.218 —
  above both observed seeds and above the baseline's best seed (0.2124). Implausible → no 3rd seed
  spent. Cautious masking helps a hair but isn't a needle-mover here: the AdamW groups (backbone +
  det head + norms) are already well-conditioned under the horizon-30 schedule, and the per-step lever
  that *did* move this screen (Muon) operates on the enc/dec matrices the cautious mask leaves
  untouched. The C-Muon follow-up (mask the Muon group's momentum∘grad pre-Newton-Schulz) is the only
  remaining cautious variant — left in the backlog, low prior after this tie. Pivot to **Tier-1 #3
  (Moonlight RMS-match)** then **#4 (Muon-group WD)** — both reshape the *Muon* update, the active
  lever. Segment: optimizer-side only; rejected → nothing lands on trunk, no segment impact to verify.
- Env note: `uv run` silently bumped torch 2.9.0→2.9.1 mid-session; restored the pinned lock + synced
  the venv back to 2.9.0 before the next runs so #3/#4 keep baseline parity. The tie is robust to a
  torch patch bump regardless.

## 2026-06-13 — pmc (Stable-DINO PMC, matcher cost modulation)   [rejected — tie]
- Paper / source: "Detection Transformer with Stable Matching" (arXiv:2304.04742, ICCV'23) PMC;
  corroborated by Rank-DETR high-order cost (arXiv:2310.08854). ideas.md Tier-1 #1.
- Hypothesis: modulate the class prob inside the matching cost by overlap, `p ← p·((GIoU+1)/2)^0.5`,
  to kill the confident-but-misplaced-query-steals-GT failure and matching churn. On the 55% of
  objects < 16 px (IoU near-binary, L1 flat) the score-driven class cost dominates pair ranking —
  exactly what PMC is meant to fix. Train-only (matcher is `@torch.no_grad`), zero latency/TRT risk.
- Change (files): `dfine_seg/model/matcher.py` — hoisted pairwise GIoU above the class cost, modulated
  `out_prob` in the focal branch by `((giou+1)/2).clamp(0,1).pow(0.5)`, reused giou for `cost_giou`.
  ~6 lines. exp/pmc sha `d392c80`. `make test` 89/89.
- Result (test, 2 seeds): mAP_50_95 **0.2122±0.0014** (seeds .2108/.2135, gain **+0.0003**, ≪ 0.003
  margin), f1 **0.557±0.0** (TRT row, gain +0.0005, within margin), lat trt 2.1 / torch 13.5 ms
  (ratio 1.0), params 10.302M. 🔴 KEEP BEST. (TRT bench row healthy — no export regression.)
- Read: clean **tie**, not a regression — both metrics nudge up a hair but nowhere near the margin.
  The 12-epoch COCO +0.4 AP did **not** transfer to this VisDrone screen. Most likely reason: D-FINE
  already carries a strong localization signal in the matching cost (`cost_bbox` 5 + `cost_giou` 2 vs
  `cost_class` 2 — geometry already outweighs class 7:2), and CDN supplies clean positive matches
  every step, so the "confident-but-misplaced query steals the GT" pathology PMC targets is largely
  pre-empted here; modulating class prob by overlap adds little on top. Seed spread 0.0027 < margin
  → no 3rd seed needed (and the mean would have to jump to >0.2149 to promote — implausible). Matcher
  cost remains a quiet surface: the cheap config follow-up (idea-1-contingent `cost_class 2→1`,
  ideas.md §12) is now **lower** prior since the class-cost *shape* change here was inert — deprioritize
  it. Pivot to the optimizer side: **Tier-1 #2 Cautious AdamW** is next (per-step update quality, the
  lever that actually moved the needle here — Muon). Segment: PMC is shared-matcher code; since it's
  rejected, nothing lands on the trunk, so no segment impact to verify.

## 2026-06-13 — baseline_h30 (horizon-30 re-baseline)   [PROMOTED — new persisted baseline]
- Source: methodology change `train.epochs` 100→30 (guide rule 9 + §8) invalidated the old
  horizon-100 bar (0.2061/0.552). Per ideas.md, re-run the current-best (Muon) recipe **unchanged**
  at horizon-30 and re-pin `baseline.json`. **maxDets validator fix NOT applied** (user decision
  2026-06-13: leave detections-per-image as is — `validator.py` stays frozen/unmodified).
- Change (files): none to code. Removed stale `baseline.json` so `promote.py` re-establishes; ran on
  `exp/baseline-h30` (sha `239b67c`, = the horizon-30 doc-setup commit), 2 seeds [42,123].
- Result (test, 2 seeds): mAP_50_95 **0.2119±0.0005** (seeds .2114/.2124), f1 **0.5565±0.0005**
  (TRT row, seeds .556/.557), lat trt 2.1 / torch 13.35 ms (ratio 1.0), params 10.302M. Walltime
  140 min total. TRT bench row healthy (f1 0.557, not 0) → export OK.
- Read: horizon-30 raises absolute numbers vs horizon-100 (mAP +0.0058, f1 +0.0045) — exactly the
  expected effect of letting the anneal mostly complete instead of stopping at ~96% of peak LR; the
  *recipe* is identical (Muon), only the LR-schedule horizon changed. Per-seed std stays tiny
  (0.0005 ≪ 0.003 floor), so the margin floor governs promotion as before. **This 0.2119/0.5565 is
  the new bar for every subsequent candidate** (first up: Stable-DINO PMC, ideas.md Tier-1 #1).

## 2026-06-10 — TRT fp16 root-cause + export hardening LANDED; qknorm_full75 confirmation; QK-norm stays shelved
- Source: user-driven deep dive into the qk-norm TRT collapse + a full 75-ep qk-norm confirmation run.
- **Part 1 — the fp16 0-detection collapse, root-caused & fixed.** Not SDPA math, not fp16 range, not
  the kernel: a **TensorRT fusion bug around GridSample in fp16**. Same full-fp16 ONNX is correct in
  onnxruntime; the full-fp16 TRT engine becomes correct the moment GridSample's edges are unfused.
  Fix landed in `dfine_seg/dl/export.py`: `half=True` → fp16 ONNX (`op_block_list=["GridSample"]`,
  `keep_io_types`) parsed into a **STRONGLY_TYPED** engine. Validated (full test set): screen qk-norm
  s42 0.0→0.552 / s123 0.0→0.549 @ 2.1 ms; **muon_full75 0.585 @ 2.1 ms = auto-FP16 parity** — and the
  pin is load-bearing for *every* model: strong-typed full-fp16 silently drops muon to **0.559** (no
  NaN!). TRT 11 removes auto-FP16 entirely → this is the only forward-compatible fp16 path. Upgrading
  TRT does NOT remove the need (10.16/11.0 still broken; 10.16 fp16 NaNs even pinned).
- **Part 2 — qknorm_full75** (COCO-init via fused-in_proj→q/k/v remap, Muon, 75 ep, seed42; exact
  muon_full75 recipe + qk_norm): training **WIN** — test mAP_50_95 **0.2388 vs 0.2359 (+0.0029)**,
  mAP_50 0.4104 vs 0.4063, ≥ muon at every decade epoch. But bench torch 0.582 vs **TRT 0.545**, and
  forensics prove a **weights-dependent TRT compiler defect at ALL precisions** (fp32 0.552, scores
  scatter ±0.5; ORT-fp32=ORT-fp16=torch on identical inputs; structurally identical ONNX vs the
  correctly-compiled screen ckpt; standalone GridSample with captured inputs maxdiff 1e-5; TF32/opt-
  level/onnxsim/explicit-attention/clamp/fp32-tails all ruled out; 10.16/11.0 fp32 = 0.529).
- Verdict: **QK-norm shelved** — undeployable on the TRT stack despite training merit. Full recipe,
  evidence and revisit conditions: `experiments/qk_norm.md`; working code on `exp/qk-norm-lr`
  (arch f45f71f, remap a5445ab); artifacts in `experiments/runs/qknorm_full75/seed42`. **Muon remains
  the deployable best.** Guard lesson re-confirmed: only the TRT-row f1 catches this class of failure.

## 2026-06-08 — qk-norm-lr (QK-norm @ baseline LR peak 0.005)   [rejected — accuracy-neutral; TRT export incompatible → SHELVED]
- Source: clean 1-change-vs-Muon follow-up to qk-norm@0.01, isolating QK-norm's accuracy effect at the
  baseline LR (peak 0.01 was stable but flat). User-directed.
- Change (files): same QK-norm arch (exp/qk-norm f45f71f) at muon_lr default → peak 0.005. exp/qk-norm-lr f8d59e6.
- Result (test, 2 seeds): mAP_50_95 0.2076±0.0016 (seeds .206/.2091, gain **+0.0015** vs 0.2061, within margin),
  PyTorch f1 0.554 — but **TensorRT f1 0.0** (fp16 export broken). lat: fp16-TRT 1.9 ms (broken engine), fp32-TRT
  3.1 ms (correct, 1.48×), torch ~13.5. params 10.303M. 🔴 KEEP BEST → SHELVED.
- Read: **QK-norm is accuracy-neutral** at both LRs (+0.0015 here, −0.0018 at 0.01 → higher LR doesn't help),
  so it's not a campaign win regardless of deployment. **TRT investigation (user-driven, all-format bench on
  seed42):** the SDPA/QK-norm graph is correct on **torch, ONNX, OpenVINO, LiteRT** (all 0.554) and converts on
  **CoreML** — **only the fp16 TensorRT engine collapses to 0 detections.** So it's a TRT fp16-build bug for the
  decomposed-SDPA + per-head-LayerNorm pattern, not a model/ONNX/fp16-general fault. fp32 TRT confirms (0.553 @
  3.1 ms) but fails the ≤1.05× latency budget with no accuracy win; a surgical op-blocklist (LayerNorm+Softmax→
  fp32) did **not** restore the fp16 engine. Per simplicity + latency rules → **shelve** (code kept on exp branches).
  **Payoff of the whole QK-norm arc:** (1) root-caused the DETR-family instability (unbounded box-corner logits +
  attention-logit growth; YOLO avoids it via BN everywhere + bounded anchor-relative boxes + fixed assignment);
  (2) QK-norm is a proven *torch-side* fix for the issue-#64 NaN class if ever needed; (3) **methodology fix** —
  the f1 guard now reads the **TensorRT** bench row (it silently passed a 0-detection export twice on the old
  PyTorch-row guard). **Lesson for next agents: for any change that alters the export graph (SDPA / new ops /
  arch), verify trt_f1 ≈ torch_f1 — a healthy torch model can still produce a dead TRT engine.**

## 2026-06-08 — qk-norm (per-head QK-LayerNorm on enc/dec self-attn, @ peak 0.01)   [rejected — STABILITY SOLVED, accuracy flat]
- Paper / source: QK-Norm (Dehghani et al. ViT-22B; Chameleon). The stability fix for the muon-lr NaN.
- Hypothesis: bound the attention logits (the structural analogue of YOLO's BN) so the peak-0.01 regime
  that NaN'd can train, and Muon's LR-transfer can pay off once the basin is stable.
- Change (files): `QKNormSelfAttention` (arch/utils.py) — LayerNorm Q,K per head before the SDPA
  dot-product, drop-in for nn.MultiheadAttention on enc + dec **self**-attn (cross-attn is deformable,
  untouched). Gated by `build_model(qk_norm=...)` ← `config.yaml train.qk_norm` (default off, threaded to
  all 5 build sites); self-describing (q_norm keys → Torch_model auto-detects, crosses frozen bench.py);
  decoder denoising bool mask → SDPA additive −inf. exp/qk-norm f45f71f. This run = qk_norm + muon_lr=0.005
  (peak 0.01) → **2 changes** vs Muon. make test 76/76 (+2 new: shapes/grad + mask-convention).
- Result (test, 2 seeds): mAP_50_95 0.2043±0.0011 (seeds .2055/.2032, gain −0.0018 vs 0.2061, within
  margin), f1 0.551±0.0 (gain −0.001), lat trt 1.85 ms (ratio 0.88, **faster**), params 10.303M. 🔴 KEEP BEST.
- Read: **The stability question is answered — QK-norm fixes it.** Both seeds ran clean to the walltime
  cap with zero NaN, where muon-lr (same peak 0.01, no QK-norm) NaN'd at ep16. Latency even improved (the
  SDPA path beats nn.MultiheadAttention's fused kernel here). Mid-run val tracked *higher* than muon-lr's
  pre-NaN val at matched epochs (ep15 0.239 vs 0.227) — so QK-norm helps the fit — but the final test mAP
  lands at the Muon baseline: **at peak 0.01 the 2× LR, even stabilized, buys no net accuracy.** This is
  the user's predicted branch (stable, higher-LR-flat). Confound: 2 changes (qk_norm + LR). Next:
  `qk-norm-lr` = QK-norm @ baseline peak 0.005 (one change vs Muon) to isolate QK-norm's accuracy effect at
  the original LR — the clean, promotable comparison. If flat there too, QK-norm is a robustness win for
  real users (issue #64 — it removes the NaN-divergence failure class) rather than a VisDrone accuracy gain.

## 2026-06-08 — muon-lr (Muon peak-LR retune to 0.01)   [FAILED — NaN divergence]
- Paper / source: Muon LR-transfer band ~0.01–0.02 (Moonlight, arXiv:2502.16982). ideas.md #1.
- Hypothesis: our Muon peak 0.005 sits below the robust band; doubling to peak 0.01 (`muon_lr=base_lr*20`)
  should step the enc/dec matrices more efficiently under the ~22-epoch cap.
- Change (files): exposed `train.muon_lr` (config.yaml null→base_lr*10; train.py read), research override
  0.005 → OneCycle peak 0.01. exp/muon-lr sha 33fb120. `make test` 74/74.
- Result: **seed42 diverged to NaN at epoch 16** (batch 88). Loss sat ~20 then *sudden* NaN boxes →
  `generalized_box_iou` degeneracy assert (utils.py:41) → run aborted; salvaged pre-NaN ckpt test mAP
  0.1862 < baseline 0.2061. seed123 was on the same path (killed at epoch 7 to free the GPU). 🔴 FAILED.
- Read (root cause): the box-corner distribution logits `pred_corners` (dfine_decoder.py:510) are
  **unbounded and accumulate residually across all 6 decoder layers**; only the attention `target` (:256)
  and query-pos embed (:486) are clamped, not the corner head. 2× LR drifts them up until — under fp16 AMP
  (ceiling 65504) — one logit → inf → softmax in `integral` → NaN box → matcher crash. Same class as
  issue #64 (there 226 slow epochs; here 16 at 2× LR). Sudden-NaN-at-stable-loss = overflow, not gradual
  divergence. **User: bf16 was tried before and didn't help** → not *only* the fp16 ceiling but genuine
  attention-logit/residual growth → conservative LRs are load-bearing. Why YOLO doesn't: bounded
  anchor-relative boxes (can't NaN), BN after every conv (hard-normalized, big-LR-tolerant), fixed
  assignment (no Hungarian cost over predicted geometry). Implication → test **QK-norm** (per-head
  QK-LayerNorm) to bound attention logits = give the transformer YOLO's BN; run at the same peak 0.01 to
  see if it rescues the regime.

## 2026-06-08 — muon (Muon optimizer for enc/dec 2D matrices)   [PROMOTED — first real win]
- Paper / source: Muon (Jordan et al., 2024) — Newton-Schulz-orthogonalized momentum for 2D weight
  matrices; ~faster convergence on speedruns. ideas.md #3.
- Hypothesis: after CDN (#1) and Dense O2O (#2) both *regressed* mAP, the lesson was that the
  walltime-cap bottleneck is per-step **optimization efficiency**, not supervision density. Muon
  attacks exactly that — orthogonalized updates on the high-condition-number enc/dec attention/MLP
  linears, where per-step gains compound most under ~22 epochs. Optimizer-only → zero inference latency.
- Change (files): new `dfine_seg/model/muon.py` (`MuonWithAuxAdam`, single-device, one `.step()` so the
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
- Change (files): `dfine_seg/model/configs.py:19` `num_denoising` 100→300 (1 line). exp/cdn-denoising sha `21970e4`.
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
  `dfine_seg/model/arch/hgnetv2.py` (dist-safe pretrained-backbone load), `configs/research_visdrone.yaml`
  (`train.batch_size=8`, `train.epochs=100`), `dfine_seg/dl/train.py` (fail loudly on mid-train CUDA OOM).
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
