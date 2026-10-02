"""Run one experiment: for each seed, train, export TensorRT, bench and append a row to results.tsv.

    uv run python experiments/run.py --name baseline --preset configs/research_<dataset>.yaml
    uv run python experiments/run.py --name lovasz --preset ... --abort-below 0.695  # skip seed 2

A preset is a local YAML file (keep it in the gitignored `configs/`) with one `overrides:` mapping of
Hydra keys to values, i.e. only what differs from root config.yaml (data paths, classes, epochs,
img_size, pinned batch size). Its file stem, minus a `research_` prefix, tags the results.tsv rows
and the experiments/runs/<tag>/ folder.

Whatever is checked out is the candidate. Extra `key=value` args (placed before the flags) are
appended as Hydra overrides, for smoke tests only. Campaign runs use the preset alone.
"""

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results.tsv"
COLS = ["date", "preset", "name", "branch", "sha", "dirty", "seed", "epochs", "train_min"]
COLS += ["miou_torch", "miou_trt", "lat_trt_ms"]


def hydra_val(v):
    if isinstance(v, bool):
        return str(v).lower()
    if v is None:
        return "null"
    if isinstance(v, list):
        return "[" + ",".join(map(str, v)) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{x}" for k, x in v.items()) + "}"
    return str(v)


def to_overrides(preset):
    out = []
    for k, v in preset.items():
        # Hydra can't add keys to an existing dict (e.g. more classes), so replace it whole.
        out += [f"~{k}", f"+{k}={hydra_val(v)}"] if isinstance(v, dict) else [f"{k}={hydra_val(v)}"]
    return out


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def step(module, overrides):
    cmd = ["uv", "run", "python", "-m", f"dfine_seg.dl.{module}", *overrides]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--preset", required=True, help="local preset YAML, see the module docstring")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument(
        "--abort-below", type=float, help="stop after a seed whose TRT mIoU is below this"
    )
    args, extra = ap.parse_known_args()

    preset = yaml.safe_load((REPO / args.preset).read_text())["overrides"]
    base = to_overrides(preset) + extra
    tag = Path(args.preset).stem.removeprefix("research_")
    info = {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "sha": git("rev-parse", "--short", "HEAD"),
        "dirty": int(bool(git("status", "--porcelain", "--untracked-files=no"))),
    }

    for seed in args.seeds:
        out = REPO / "experiments" / "runs" / tag / args.name / f"seed{seed}"
        ov = [*base, f"train.path_to_save={out}", f"exp_name={args.name}_s{seed}"]
        t0 = time.time()
        step("train", [*ov, f"train.seed={seed}"])
        train_min = (time.time() - t0) / 60
        step("export", ov)
        step("bench", ov)

        bench = pd.read_csv(out / "bench_metrics.csv", index_col=0)
        row = {
            "date": datetime.now().isoformat(timespec="minutes"),
            "preset": tag,
            "name": args.name,
            **info,
            "seed": seed,
            "epochs": preset.get("train.epochs"),
            "train_min": round(train_min, 1),
            "miou_torch": pd.read_csv(out / "metrics.csv", index_col=0).loc["val", "mIoU"],
            "miou_trt": bench.loc["TensorRT", "mIoU"],
            "lat_trt_ms": bench.loc["TensorRT", "latency"],
        }
        new = not RESULTS.exists()
        with RESULTS.open("a", newline="") as f:
            w = csv.DictWriter(f, COLS, delimiter="\t")
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"\nRESULT {row}", flush=True)
        if args.abort_below is not None and row["miou_trt"] < args.abort_below:
            print(
                f"ABORT: TRT mIoU {row['miou_trt']} < {args.abort_below}, skipping remaining seeds"
            )
            break


if __name__ == "__main__":
    main()
