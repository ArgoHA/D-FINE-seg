# Sem-seg autoresearch loop

**Goal:** a better D-FINE-seg semantic-segmentation model. Better means higher TensorRT fp16 mIoU at
the same TensorRT latency. Changes should be general, not Cityscapes tricks: we care about transfer
to off-road driving. Cityscapes is the screen and the off-road set is the held-out check.

Read `AGENTS.md`, `experiments/HISTORY.md`, `experiments/ideas.md` and `experiments/results.tsv` first.

## Setup (fixed for the whole campaign)
- Presets are local and not in git (`configs/`, gitignored): the Cityscapes screen preset and the
  off-road preset. `run.py`'s docstring describes the format.
- `experiments/run.py --name <slug> --preset <preset>` trains 3 seeds (42, 123, 7)
  with a fixed epoch count, then exports and benches TensorRT for each seed and appends one row per
  seed to `results.tsv`. Presets pin everything else. Never edit a preset, `run.py`,
  `dfine_seg/dl/validator.py` or `dfine_seg/dl/bench.py` to change a result.
- The baseline is the current `exp_base` row in `results.tsv`. Re-run it only after the preset or
  hardware changes.
- Branches are local only and never pushed. `exp_base` is the current best. Each experiment runs on
  `exp/<slug>`, branched from `exp_base`, and gets one commit. A winner is fast-forwarded into
  `exp_base`. A loser's branch is kept for reference.

## Decision (per experiment, mean of 3 seeds, TRT row)
The same config and seed rerun differs by about 0.005 mIoU (GPU nondeterminism), so single runs prove nothing.
Keep the candidate if all of these hold:
- ΔmIoU ≥ 0.006
- latency ≤ 1.03× the baseline
- the extra complexity is justified: a big change for a small win is a reject

A candidate that is at least 5% faster with ΔmIoU ≥ −0.001 is also a win. If one seed is clearly
worse (both metrics past the threshold), skip the remaining seeds and reject.

## Research
Read the code (`dfine_seg/model/`, `dfine_seg/dl/train.py`, the sem_seg dataset and loss), the
results so far and recent real-time segmentation papers. Append 5–10 ideas to `ideas.md`, ranked by
expected gain divided by implementation cost. For each idea, give: what to change, which files, why it
should help, the latency risk and a source. One idea is one isolated change. Prefer changes scoped to
`task == sem_seg`, and flag any that touch shared detection or instance-segmentation code. Don't
repeat anything in `HISTORY.md` or `ideas.md` unless you name what is different this time.

## Execute
Take the top `pending` idea and repeat:
1. `git checkout -b exp/<slug> exp_base`. A **subagent** implements the change, runs
   `uv run ruff format . && uv run ruff check . && make test-fast`, and commits. Keep your own context
   for orchestration.
2. Run `uv run python experiments/run.py --name <slug> --preset <Cityscapes preset>` in the background and wait for its completion
   notification instead of polling.
3. Decide. Update the idea in `ideas.md` with its status (✅ / ❌ / 💥 failed), numbers and a
   one-line reason. On a win, run `git checkout exp_base && git merge --ff-only exp/<slug>`.
4. Send a notification with `uv run python experiments/notify.py "<verdict, numbers, next idea>"`.
5. After every 5 experiments where `exp_base` moved: run `run.py` with the off-road preset
   on `exp_base` and compare against the off-road baseline row. Report whether the gain transfers.

Stop when no pending ideas remain or when the user says so. Run detection and instance-segmentation
regression tests (`scripts/regression_test.py`) only when the user asks.
