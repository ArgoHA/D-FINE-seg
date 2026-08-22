# EXPERIMENT_GUIDE.md — autoresearch playbook

How an AI agent runs the D-FINE-seg improvement loop. Read this together with `CLAUDE.md` (repo mechanics) before starting. Follow it literally.

## 0. Mission
Improve **D-FINE-seg itself** — the model + training (detection and segmentation; the campaign trains `detect`, segmentation must not regress — rule 10) without meaningfully raising inference latency. **VisDrone is only the experimentation dataset**: a fast, hard screen (small/dense objects, ~1h to train), a proxy — not the target. Prefer changes whose *mechanism* is general over dataset-specific tuning.

## 1. Hard rules (do not violate)
1. **One change per experiment.** Isolate the variable, or the ledger is meaningless.
2. **Init = ImageNet backbone only:** `train.imagenet_backbone=true` + `train.pretrained_model_path=null` (set in `research_visdrone.yaml`). Never load `dfine_*_coco.pt` — biases toward the current arch. The backbone is **always pretrained**; swapping in a different backbone means bringing *its* pretrained (ImageNet) weights too, so comparisons stay fair.
3. **Frozen files — never edit to change a result:** `dfine_seg/dl/validator.py`, `dfine_seg/dl/bench.py`, `scripts/run_candidate.py`, `scripts/promote.py`. These define how success is measured. `promote.py` rejects any candidate whose diff touches them.
4. **Edit the model and the training process:** `dfine_seg/model/` (arch, losses, matcher) and `dfine_seg/dl/train.py` **when the change optimizes training** (optimizer, schedule, EMA, augmentation, loss wiring). Pure-config ideas go in `configs/research_visdrone.yaml` overrides.
5. **Latency budget:** promote only if latency ≤ 1.05× baseline, OR ≤ 1.20× with a *2× margin* accuracy win. Enforced by `promote.py`.
6. **Complexity / simplicity rule.** If a change adds a lot of code or large structural complexity and the accuracy/latency delta is marginal, **skip it.** Prefer the simpler model. A borderline metric win does not justify a big, hard-to-maintain change — say so in the notebook and keep the baseline.
7. **`make test` must pass** before training (full suite, ~6s). If it fails, the change is broken — fix or abandon, don't train.
8. **Approval gate (interactive mode):** after research, present a short proposal and WAIT for the user to approve / edit / skip before implementing and burning GPU time. (See §7 for when this is relaxed.)
9. **Fixed campaign constants — do not retune.** `train.epochs=30` and `harness.seeds` are held constant across the whole campaign. Never change them *to chase a result*; if you ever must, re-baseline. Seeds are `[42, 123]`.
10. **Don't harm the `segment` variant.** The campaign trains `detect`, but any change that lands as model code or a shared-config default also runs on `segment`. A change that helps detect must not regress segmentation. If a change is unsafe for masks or touches the mask head, gate it on `task` / keep it a detect-only override in `research_visdrone.yaml`, and note the segment impact in the notebook.

## 2. Branching model
- `main` — the user's real project. Never commit experiments here.
- `main_exp` — the **single durable trunk**: harness + ledger + notebook + ideas + baseline + current-best model code. Created once from `main` + the harness commit. Everything research lives here, so `main` stays untouched.
- `exp/<name>` — one per experiment, branched from `main_exp`.

Promotion = fast-forward `main_exp` to the winning `exp/<name>` (it carries both the code change and that run's ledger/notebook/baseline commits). On rejection, commit only the ledger/notebook update straight to `main_exp` and leave `exp/<name>` in place for forensics. Either way the tracking files advance on `main_exp` every iteration — that is what lets a fresh agent resume.

## 3. The decision (in `promote.py`)
Two metrics, both on the held-out **test** set, mean over seeds:
- **mAP_50_95** — from training `metrics.csv` (test row). Bench mAPs are meaningless (bench runs at a single conf threshold), so mAP always comes from training.
- **f1** — from `bench_metrics.csv` (**TensorRT row**) — the *actual deployment artifact* (TRT engine + NMS), so the no-regression floor also catches a **broken or degraded export**: a change can train fine in PyTorch yet produce a TRT engine that collapses.

Both count toward promotion via their **average gain**; neither may regress past its own margin.

```
gain_map  = cand_map - best_map ;  gain_f1 = cand_f1 - best_f1
avg_gain  = (gain_map + gain_f1) / 2 ;  M = (map_margin + f1_margin) / 2
margin    = current best's across-seed std for that metric (floor 0.003)
lat_ratio = cand_latency / best_latency      (TensorRT, fallback PyTorch)
PROMOTE if  gain_map > -map_margin  and  gain_f1 > -f1_margin            (neither regresses)
       and  ( (avg_gain > M    and lat_ratio <= 1.05)
              or (avg_gain > 2*M and lat_ratio <= 1.20) )
```
`margin` is the variance we actually measured, so a "win" must clear real noise. The simplicity rule (1.6) still overrides a marginal pass.

**Export sanity check:** `run_candidate.py` flags any seed whose **torch vs TRT f1 differ by > 0.003** as a likely-broken/degraded export (warn-only, in the run log + `trt_export_flagged` in the result JSON) — an independent catch beyond the f1 no-regression floor.

## 4. The baseline is established ONCE and persisted
`experiments/baseline.json` describes the **current best**. It is created on the very first run and then overwritten automatically whenever a candidate is promoted.

- **First run ever** (no `baseline.json`): your first experiment is the **unchanged** architecture — the control. `promote.py` detects the missing file, stores it as the baseline, and logs it. Commit it. This 2-seed run happens **exactly once in the project's life.**
- **Every later agent/session**: `baseline.json` already exists and is committed → **never re-train the control.** Read it and go straight to the next idea.

Establish it (one time):
```bash
git checkout -b exp/baseline main_exp
uv run python scripts/run_candidate.py --name baseline --comment "control"
uv run python scripts/promote.py --candidate experiments/runs/baseline/candidate_result.json
git branch -f main_exp HEAD            # control is the first 'best'
# commit experiments/baseline.json + ledger.csv + notebook on main_exp
```

## 5. The loop (one iteration)
**A. Bootstrap / read state.** Be on the trunk: `git checkout main_exp`. Read this guide,
`experiments/lab_notebook.md` (start with the
**Current state** block at the top), `experiments/ledger.csv`, `experiments/ideas.md`, and
`baseline.json`. This tells you the current best, what's been tried, and what's next — so a fresh
agent picks up exactly where the last one stopped.

**B. Research.** Pick the top item from `ideas.md` (or research a new one). Read the actual paper
(WebSearch / WebFetch). Read the relevant code: `dfine_seg/model/arch/`, `dfine_seg/model/configs.py`, the
loss/matcher in `dfine_seg/model/`, and `dfine_seg/dl/train.py` for training-process ideas.

**C. Propose (STOP for approval in interactive mode).** Present, in ~10 lines: the idea + paper, the
exact change and files, why it should help convergence/accuracy, latency risk, complexity cost, and
expected result. Wait for approve / edit / skip.

**D. Implement.** `git checkout -b exp/<name> main_exp`. Make the single change. `make test` — must
pass. Commit.

**E. Run candidate** (≈2×60 min train + per-seed export/bench):
Launch in the **background** redirecting to the `current.log`;
```bash
nohup uv run python scripts/run_candidate.py --name <name> --comment "<what changed>" \
  > experiments/runs/current.log 2>&1 &
# user: `tail -f experiments/runs/<name>.log` to watch epochs live.
# agent: poll experiments/runs/<name>.log + train_log.txt.
```
Full/long manual runs (§6) follow the same background pattern.

**Seed-1 early abort (save GPU on clear losers):** if the first seed ends with *both* mAP_50_95 and f1 > 0.002 below baseline, kill the run, skip the remaining seed(s), and reject (keep best) — don't spend a second seed on a change already losing on both metrics.

**F. Decide + log.**
```bash
uv run python scripts/promote.py --candidate experiments/runs/<name>/candidate_result.json --base main_exp
```
Then update **BOTH** tracking docs — required on **every** terminal outcome (promote, reject, fail):
- `lab_notebook.md`: the **Current state** block AND a dated entry (why it worked or didn't —
  required for rejections too).
- `ideas.md`: record **what was tried** on the idea you just ran — mark it 🔴/🟢 with the one-line
  result + a pointer to the notebook, and update the "next up" pointer. You do **not** need to add
  *new* ideas, but the run-queue must never still list a tried idea as pending (a fresh agent would
  re-run it). On promotion you additionally move/retire the idea (step G).

Commit ledger + notebook + ideas.md (+ baseline.json if changed) on `main_exp` (on rejection) or on
the `exp/<name>` branch (so a promotion carries them).

**Then notify the user** (run after `promote.py`, so the ledger holds the verdict):
```bash
uv run python scripts/notify.py --name <name>     # sends verdict + metrics to Telegram
```
Do this on **every** terminal outcome — promote, reject, *and* a failed/aborted run (use
`--message "baseline OOM'd, investigating"` for the failure case). Creds come from `.env`
(`TG_TOKEN` / `TG_CHAT_ID`, git-ignored); missing creds only warn, never block the run.

**G. Promote if green.** Only if the verdict is 🟢:
```bash
git branch -f main_exp HEAD
```
Move the idea out of `ideas.md`. (`promote.py` already updated `baseline.json` to this candidate.)

## 6. Full / COCO confirmation — manual, and only unbiased for non-arch changes
VisDrone (ImageNet-init, 60-min) is the fast **screen**; the real decision to run a longer/full
training is **manual** — the agent never auto-launches one. The promotion loop (§3) stays on the
2-seed screen; a full run is a human-triggered production check, not part of the automatic verdict.

**The COCO-init bias rule (read before proposing any full run).** A full run that inits from
`dfine_<size>_coco.pt` (or any COCO-pretrained checkpoint) and compares against the COCO-pretrained
reference (`det_s_2026-02-22`: test mAP_50_95 0.2316 / f1 0.5621) is **only unbiased when the
candidate's architecture is byte-identical to that reference** — i.e. the change is optimizer /
schedule / augmentation / loss-wiring (e.g. Muon). Then the same COCO weights load into the same
graph and only the *training process* differs → fair.

For **architecture changes** (new/modified layers) a COCO-init comparison is biased — the unchanged
layers get a free COCO head-start while the novel layers start cold against a fully-warm reference.
There is no cheap unbiased COCO number. The fair options are: (a) full **ImageNet-init** runs of
*both* the new and current-best architectures (share the init; abandon the COCO reference as the bar,
accept lower absolute numbers), or (b) actually COCO-pretrain the new architecture first — the
expensive, correct step, done only once you commit to adopting it. In-loop, judge architecture ideas
on the 2-seed screen and defer COCO validation to real adoption. Flag VisDrone-specific tuning (e.g.
small-object matcher hacks) as likely-non-transferring in the notebook.

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
  ledger + notebook. **Send the Telegram notification (§5.F) after every experiment** so the user can
  follow progress remotely without watching the box.

## 8. Gotchas
- **Walltime governs *when we stop*, but `epochs` sets the LR-schedule horizon.** `train.epochs=30`
  (fixed — **do not change**; 100 until 2026-06-13, originally 1000), `train.max_walltime_min=60`.
  60 min ≈ 21 epochs, so runs stop at ~epoch 20-25 — i.e. ~65-80% through the 30-epoch anneal,
  ending at ~8-30% of peak LR. Why 30: at horizon 100 every run ended at ~96% of peak LR (45%
  warmup / 55% cruise / 0% decay) — an operating point full training never produces, so screen
  verdicts were measured in an unrealistic state; horizon 30 keeps warmup short (3 epochs) and lets
  the anneal mostly complete. At `epochs=1000` the run never leaves warmup (starved convergence);
  never go back. `epochs` is **not** a "train this long" knob — walltime ends the run. It is a
  campaign constant like the seeds: changing it re-shapes every run's LR and invalidates the
  baseline margin — re-baseline when changed (**horizon-30 re-baseline: PENDING**). Stop-epoch
  jitter now maps to end-LR jitter (~8-30% of peak); the margin floor absorbs it — the price of a
  realistic end state. Mosaic-close is pinned OFF in the research config (`no_mosaic_epochs: 0`,
  user decision 2026-06-13): the default close at `epochs-5`=25 could fire on a fast seed but not a
  slow one — pinning keeps seeds comparable. "No clean-data tail" is an accepted campaign constant.
- **Accuracy split = test.** Both f1 (bench, **TensorRT row** — §3) and mAP_50_95 (train) are read from
  the test set; keep it that way so candidate and baseline are comparable.
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
