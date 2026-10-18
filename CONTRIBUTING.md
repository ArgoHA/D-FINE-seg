# Contributing to D-FINE-seg

D-FINE-seg is maintained by one person and used daily for production training work, so it is steered deliberately: few features, little code, and no regressions in accuracy or latency. Please read this page before opening an issue or a pull request.

## Getting help

Ask questions in [GitHub Issues](https://github.com/ArgoHA/D-FINE-seg/issues). Check the [README](README.md) and existing issues first.

## Reporting bugs

Open an issue with the **Bug report** template and include:

- `dfine-seg` version, install method (pip or source), OS, GPU, and the versions of torch and of the backend in use (TensorRT, OpenVINO, ...);
- the task (`detect`, `segment`, `sem_seg`), model size, and the relevant config;
- the command you ran and the full error or a description of the wrong output;
- a minimal reproduction, if you can share one.

## Proposing changes

**Open an issue before writing code.** Use the **PR proposal** template (it adds the `pr-proposal` label): describe the problem, how you would solve it, and what it touches. We agree on the design in the issue, and only then do you open a pull request. Pull requests for features nobody has discussed might be closed.

Typo fixes, documentation fixes, and small, obvious bug fixes can go straight to a pull request.

Before proposing, keep in mind:

- Less code is preferred. Tiny accuracy gains rarely justify new code, options, or complexity.
- Changes must respect the project invariants in [`AGENTS.md`](AGENTS.md). They apply to people as well as coding agents.
- An accepted idea may be reimplemented by the maintainer to fit the codebase. If so, you are credited as co-author of the commit.

## Pull requests

- Keep the pull request limited to the agreed scope and link the issue.
- Run `uv run ruff format . && uv run ruff check . && make test-fast`. If model outputs change, also run `make test`.
- If the change can affect training, export, inference, accuracy, or latency, attach a before/after comparison: the same config on a dataset of your choice, with `dfine bench` metrics and latency from `main` and from your branch. The maintainer runs the full Cityscapes regression test (`scripts/regression_test.py`) before merging.
- **AI-assisted code is welcome, but you own it.** State in the pull request which AI tools you used and for what. Have the diff reviewed before submitting (for example, in a fresh session or by a different model), and review it yourself. You must understand how your change works and be able to answer review questions about it.
