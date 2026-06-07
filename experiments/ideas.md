# Idea backlog — candidate improvements

Prioritized hypotheses to try. **Research the actual paper before implementing** — these are
pointers, not verified claims. Move an idea to `lab_notebook.md` once tried. Bias toward ideas that
(a) help *convergence* (we only train ~60 min) and (b) are latency-neutral at inference. Keep
`verify on COCO later` in mind: prefer changes whose mechanism should generalize beyond VisDrone's
small/dense objects.

## Tier 1 — fast-convergence / latency-neutral (best fit for the walltime cap)
- **DEIM Dense O2O (heavy mosaic), CVPR 2025.** Dense O2O packs more objects per training image (via
  mosaic-style aug) → many more one-to-one positive matches per step → denser supervision, ideal under
  the walltime cap. Inference graph unchanged → latency-neutral. Touches the dataloader/aug config
  (`mosaic_augs` in the config) + possibly `train.py`. **NOTE: mind CLAUDE.md gotcha #6** (mosaic is
  discouraged for *segment*; we're on detect so OK) and GUIDE §8 (mosaic-close schedule never reaches
  its end under the cap — fine, consistent across runs).
  - The **MAL** half of DEIM was tried alone (2026-06-07) and **rejected as a near-tie** — see lab
    notebook. MAL is designed to handle the low-quality matches Dense O2O introduces, so it likely only
    pays off *with* Dense O2O. After Dense O2O lands, consider re-testing **MAL + Dense O2O** together
    (the loss code is on branch `exp/mal`).
- **Muon optimizer (Jordan et al., 2024).** Orthogonalizes the momentum update for 2D weight
  matrices (Newton–Schulz), giving notably faster convergence per step in LLM/nanoGPT speedruns —
  exactly the lever that matters under a 60-min cap. Plausibly a strong fit here: the
  encoder/decoder transformer is full of 2D linear weights. Caveats to respect: Muon applies to 2D
  hidden matrices only; norms, biases, and the detection head should stay on AdamW, so it needs a
  **hybrid optimizer** (the repo already builds custom param groups in `build_optimizer`, so wiring
  is feasible). Inference cost unchanged. Evidence outside LLMs is still thin → exploratory but
  high-upside; tune the Muon LR separately. Touches `src/dl/train.py` / `build_optimizer` (allowed
  under rule 4). Latency-neutral by construction.
- **One-to-many auxiliary supervision / Group-DETR-style extra queries** during training only
  (dropped at export) — more gradient per image, zero inference cost.
- **EMA / warmup / LR-schedule tuning.** Cheap, mostly config; with random neck/head and 60 min the
  schedule shape matters a lot. Establish a strong tuned control first.

## Tier 2 — loss / assignment
- Varifocal / quality-focal refinements to the VFL classification target.
- IoU-aware or distribution-focal regression tweaks (D-FINE already uses FDR/FGL — look for
  orthogonal gains, not duplicates).
- Matcher cost reweighting for small/dense objects (VisDrone-specific — flag as possibly
  non-transferring to COCO).

## Tier 3 — architecture (watch the latency budget carefully)
- Encoder/neck feature-fusion tweaks (cheaper or stronger cross-scale fusion).
- Activation / normalization swaps in encoder/decoder.
- Query-selection method (`query_select_method` in `configs.py`) alternatives.

## Notes / constraints
- Init is **ImageNet backbone only** — backbone-heavy ideas start from pretrained HGNetv2; neck/head
  ideas start random. Account for that when reading results.
- Anything that adds inference params/FLOPs must clear the *larger* accuracy margin (see the
  decision function) AND justify its complexity (simplicity rule) to be worth the latency.
