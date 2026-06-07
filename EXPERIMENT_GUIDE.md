# EXPERIMENT_GUIDE.md — autoresearch playbook

How an AI agent runs the D-FINE-seg improvement loop. Read this together with `CLAUDE.md` (repo
mechanics) before starting. Follow it literally.

## 0. Mission
Improve detection accuracy on **VisDrone** (hard, small/dense objects, ~1h to train) **without
meaningfully raising inference latency**, via changes that should also help COCO. Each accepted
change is a small, motivated, paper-grounded edit — proven to beat the current best across seeds.

## 1. Hard rules (do not violate)
1. **One change per experiment.** Isolate the variable, or the ledger is meaningless.
2. **Init = ImageNet backbone only.** `train.imagenet_backbone=true` +
   `train.pretrained_model_path=null`. Never load `dfine_*_coco.pt` — it biases toward the original
   architecture. (Already set in `configs/research_visdrone.yaml`.)
3. **Frozen files — never edit to change a result:** `src/dl/validator.py`, `src/dl/bench.py`,
   `scripts/run_candidate.py`, `scripts/promote.py`. These define how success is measured.
   `promote.py` rejects any candidate whose diff touches them.
4. **Edit the model and the training process:** `src/d_fine/` (arch, losses, matcher) and
   `src/dl/train.py` **when the change optimizes training** (optimizer, schedule, EMA, augmentation,
   loss wiring). Pure-config ideas go in `configs/research_visdrone.yaml` overrides. Nothing else —
   and never the frozen eval files above.
5. **Latency budget:** promote only if latency ≤ 1.05× baseline, OR ≤ 1.20× with a *2× margin*
   accuracy win. Enforced by `promote.py`.
6. **Complexity / simplicity rule.** If a change adds a lot of code or large structural complexity
   and the accuracy/latency delta is marginal, **skip it.** Prefer the simpler model. A borderline
   metric win does not justify a big, hard-to-maintain change — say so in the notebook and keep the
   baseline.
7. **`make test` must pass** before training (full suite, ~6s). If it fails, the change is broken —
   fix or abandon, don't train.
8. **Approval gate (interactive mode):** after research, present a short proposal and WAIT for the
   user to approve / edit / skip before implementing and burning GPU time. (See §7 for when this is
   relaxed.)

## 2. Branching model
- `main` — the user's real project. Never commit experiments here.
- `main_exp` — the **single durable trunk**: harness + ledger + notebook + ideas + baseline +
  current-best model code. Created once from `main` + the harness commit. Everything research lives
  here, so `main` stays untouched.
- `exp/<name>` — one per experiment, branched from `main_exp`.

Promotion = fast-forward `main_exp` to the winning `exp/<name>` (it carries both the code change and
that run's ledger/notebook/baseline commits). On rejection, commit only the ledger/notebook update
straight to `main_exp` and leave `exp/<name>` in place for forensics. Either way the tracking files
advance on `main_exp` every iteration — that is what lets a fresh agent resume.

## 3. The decision (in `promote.py`)
Two metrics, both on the held-out **test** set, mean over seeds:
- **mAP_50_95** — from training `metrics.csv` (test row). **Primary.** Bench mAPs are meaningless
  (bench runs at a fixed conf threshold), so mAP always comes from training.
- **f1** — from `bench_metrics.csv` (PyTorch row). **Guard.** This is the real deployment path
  (letterbox + NMS), so f1 always comes from bench.

```
margin    = current best's across-seed std for that metric (floor 0.003)
gain_map  = cand_map - best_map ;  gain_f1 = cand_f1 - best_f1
lat_ratio = cand_latency / best_latency      (TensorRT, fallback PyTorch)
PROMOTE if  gain_f1 > -f1_margin             (f1 not regressing beyond noise)  AND
            ( (gain_map > map_margin   and lat_ratio <= 1.05)
              or (gain_map > 2*map_margin and lat_ratio <= 1.20) )
```
`margin` is the variance we actually measured, so a "win" must clear real noise. The simplicity
rule (1.6) still overrides a marginal pass.

## 4. The baseline is established ONCE and persisted
`experiments/baseline.json` describes the **current best** (not the original control). It is created
on the very first run and then overwritten automatically whenever a candidate is promoted.

- **First run ever** (no `baseline.json`): your first experiment is the **unchanged** architecture
  — the control. `promote.py` detects the missing file, stores it as the baseline, and logs it.
  Commit it. This 3-seed run happens **exactly once in the project's life.**
- **Every later agent/session**: `baseline.json` already exists and is committed → **never re-train
  the control.** Read it and go straight to the next idea. This is why starting over is not 3h.

Establish it (one time):
```bash
git checkout -b exp/baseline main_exp
uv run python scripts/run_candidate.py --name baseline --comment "control"
uv run python scripts/promote.py --candidate experiments/runs/baseline/candidate_result.json
git branch -f main_exp HEAD            # control is the first 'best'
# commit experiments/baseline.json + ledger.csv + notebook on main_exp
```
Inspect the per-seed std in `baseline.json` margins: if it's large relative to the gains you hope
for, raise seeds or walltime before trusting any single comparison.

## 5. The loop (one iteration)
**A. Bootstrap / read state.** Be on the trunk: `git checkout main_exp`. Read this guide,
`experiments/lab_notebook.md` (start with the
**Current state** block at the top), `experiments/ledger.csv`, `experiments/ideas.md`, and
`baseline.json`. This tells you the current best, what's been tried, and what's next — so a fresh
agent picks up exactly where the last one stopped.

**B. Research.** Pick the top item from `ideas.md` (or research a new one). Read the actual paper
(WebSearch / WebFetch). Read the relevant code: `src/d_fine/arch/`, `src/d_fine/configs.py`, the
loss/matcher in `src/d_fine/`, and `src/dl/train.py` for training-process ideas.

**C. Propose (STOP for approval in interactive mode).** Present, in ~10 lines: the idea + paper, the
exact change and files, why it should help convergence/accuracy, latency risk, complexity cost, and
expected result. Wait for approve / edit / skip.

**D. Implement.** `git checkout -b exp/<name> main_exp`. Make the single change. `make test` — must
pass. Commit.

**E. Run candidate** (≈3×60 min train + per-seed export/bench):
```bash
uv run python scripts/run_candidate.py --name <name> --comment "<what changed>"
```

**F. Decide + log.**
```bash
uv run python scripts/promote.py --candidate experiments/runs/<name>/candidate_result.json --base main_exp
```
Then update `lab_notebook.md`: the **Current state** block AND a dated entry (why it worked or
didn't — required for rejections too). Commit ledger + notebook (+ baseline.json if changed) on
`main_exp` (on rejection) or on the `exp/<name>` branch (so a promotion carries them).

**G. Promote if green.** Only if the verdict is 🟢:
```bash
git branch -f main_exp HEAD
```
Move the idea out of `ideas.md`. (`promote.py` already updated `baseline.json` to this candidate.)

## 6. COCO confirmation — manual only
VisDrone is the fast proxy; the real goal is COCO-transferable gains. There is **no automatic COCO
testing** in the loop. Occasionally the current `main_exp` should be confirmed on a longer COCO run
but the **user triggers that manually** when they want it. Prefer ideas with a general mechanism;
flag VisDrone-specific tuning (e.g. small-object matcher hacks) as likely-non-transferring in the
notebook so it's an obvious candidate for a manual COCO check.

## 7. Modes & continuity across agents/sessions
All state lives in committed files on `main_exp` (`ledger.csv`, `lab_notebook.md` incl. Current
state, `ideas.md`, `baseline.json`) plus the `main_exp` pointer. Any agent can stop after step F/G;
the next agent (same or different) bootstraps from §5.A and continues. Nothing is held only in chat.

Two modes:
- **Interactive (default).** Do **one** iteration, honoring the approval gate (5.C), then **stop**
  and report. The user continues later, or hands off to another agent.
- **Autonomous (overnight).** Only when the user explicitly says so (e.g. "run experiments back to
  back until I stop"). Then skip the per-experiment approval gate, pick the top `ideas.md` item each
  time, and loop steps B→G sequentially, committing after each. Keep going until the idea backlog is
  exhausted or the user returns. Still obey every hard rule (one change, frozen files, `make test`,
  latency budget, simplicity). Record every experiment so the morning review is just reading the
  ledger + notebook.

## 8. Gotchas
- **Walltime governs, not epochs.** `train.epochs=1000`, `train.max_walltime_min=60`; training stops
  mid-schedule. Mosaic-close and any epoch-fraction schedule never reach their end — consistent
  across candidates, so fair, but every run is "early schedule." Keep it identical for all runs.
- **Accuracy split = test.** Both f1 (bench) and mAP_50_95 (train) are read from the test set; keep
  it that way so candidate and baseline are comparable.
- **Determinism:** seeds are fixed in `harness.seeds`. Don't change them mid-campaign or the
  baseline margin no longer applies — re-baseline if you do. (See the note below on why we use seeds
  rather than `cudnn_fixed`.)
- **`cudnn_fixed` is not a substitute for multiple seeds.** It removes *run-to-run* nondeterminism
  for a fixed config, but the promotion question is about *seed/init variance* — whether the
  architecture is better or this init got lucky — which only multiple seeds estimate. It also can't
  give reproducibility here anyway: the walltime cap stops at a different epoch depending on machine
  load, and deterministic kernels are slower so fewer steps fit in 60 min. Hence seeds.
- **Clean tree before branching.** Build artifacts are git-ignored; if `git status` shows stray code
  changes, resolve them first or the frozen-path check sees them.
