#!/usr/bin/env python
"""
Run one research candidate end-to-end: train N seeds (walltime-capped), then
export + bench EACH seed. Accuracy is measured on the held-out TEST set:

  - f1          : from bench (TensorRT row; fallback PyTorch only if TRT not benched) — the real
                  deployment artifact, so a broken/0 TRT export fails the guard (see EXPERIMENT_GUIDE §3).
  - mAP_50_95   : from training metrics.csv (test row) — bench mAPs are meaningless
                  because bench runs at a fixed conf threshold.
  - latency     : from bench (PyTorch + TensorRT rows), mean over seeds.

Per seed, also flags any |TRT f1 - torch f1| > 0.003 as a likely-broken/degraded export
(warn-only; recorded as trt_f1_gap / trt_export_flagged in the result JSON).

Both accuracy metrics are averaged over seeds. Writes
experiments/runs/<name>/candidate_result.json. Does NOT touch git or the ledger
(that's promote.py). Whatever is in the working tree now IS the candidate, so check
out the experiment branch before running this.

Usage:
    uv run python scripts/run_candidate.py --name baseline --comment "control run"
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
TRT_F1_GAP_TOL = 0.003  # |TRT f1 - torch f1| above this flags a likely-broken/degraded export


def hydra_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def overrides_to_args(overrides):
    return [f"{k}={hydra_val(v)}" for k, v in overrides.items()]


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def read_map(run_dir, split="test"):
    """mAP_50_95 from training metrics.csv (rows 'val'/'test')."""
    df = pd.read_csv(run_dir / "metrics.csv", index_col=0)
    if split not in df.index:
        split = "val"
    return float(df.loc[split, "mAP_50_95"])


def read_bench(run_dir):
    """(f1, lat, f1_by_backend) from bench_metrics.csv (test). f1 is the TensorRT row — the deployment
    artifact; fall back to PyTorch only if TRT was not benched at all, so a present-but-~0 TRT row
    surfaces and fails the guard (EXPERIMENT_GUIDE §3). f1_by_backend carries both rows for the
    torch-vs-TRT consistency check."""
    df = pd.read_csv(run_dir / "bench_metrics.csv", index_col=0)
    f1_by = {
        key: float(df.loc[label, "f1"])
        for label, key in (("PyTorch", "torch"), ("TensorRT", "tensorrt"))
        if label in df.index and "f1" in df.columns
    }
    f1 = f1_by.get("tensorrt", f1_by.get("torch"))
    lat = {
        key: float(df.loc[label, "latency"])
        for label, key in (("PyTorch", "torch"), ("TensorRT", "tensorrt"))
        if label in df.index and "latency" in df.columns
    }
    return f1, lat, f1_by


def count_params(model_pt):
    sd = torch.load(model_pt, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    return round(sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6, 3)


def git_info():
    def g(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "sha": g("rev-parse", "--short", "HEAD"),
    }


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    n = len(vals)
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / n if n > 1 else 0.0
    return round(m, 4), round(var**0.5, 4), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="candidate slug (also branch/run-dir name)")
    ap.add_argument("--comment", default="", help="what changed vs parent (goes to ledger)")
    ap.add_argument("--config", default="configs/research_visdrone.yaml")
    ap.add_argument("--runs-dir", default="experiments/runs")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    overrides, harness = cfg["overrides"], cfg["harness"]
    seeds = harness["seeds"]
    split = harness.get("eval_split", "test")

    run_root = REPO / args.runs_dir / args.name
    run_root.mkdir(parents=True, exist_ok=True)
    base_args = overrides_to_args(overrides)
    t0 = time.time()

    per_seed = {}
    for seed in seeds:
        sd = run_root / f"seed{seed}"
        # train
        run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.dl.train",
                *base_args,
                f"train.seed={seed}",
                f"train.path_to_save={sd}",
                f"exp_name={args.name}_s{seed}",
            ]
        )
        # export + bench this seed
        run(["uv", "run", "python", "-m", "src.dl.export", *base_args, f"train.path_to_save={sd}"])
        run(["uv", "run", "python", "-m", "src.dl.bench", *base_args, f"train.path_to_save={sd}"])
        f1, lat, f1_by = read_bench(sd)
        gap = (
            round(f1_by["tensorrt"] - f1_by["torch"], 4)
            if f1_by.get("tensorrt") is not None and f1_by.get("torch") is not None
            else None
        )
        if gap is not None and abs(gap) > TRT_F1_GAP_TOL:
            print(
                f"⚠️  seed {seed}: TRT f1 {f1_by['tensorrt']} vs torch f1 {f1_by['torch']} "
                f"(gap {gap:+}, tol {TRT_F1_GAP_TOL}) — TRT export may be broken/degraded.",
                flush=True,
            )
        per_seed[seed] = {
            "mAP_50_95": read_map(sd, split),
            "f1": f1,
            "f1_torch": f1_by.get("torch"),
            "f1_trt": f1_by.get("tensorrt"),
            "trt_f1_gap": gap,
            "lat_torch": lat.get("torch"),
            "lat_trt": lat.get("tensorrt"),
        }

    params_m = count_params(run_root / f"seed{seeds[0]}" / "model.pt")
    trt_flagged = [
        s for s in seeds if (g := per_seed[s]["trt_f1_gap"]) is not None and abs(g) > TRT_F1_GAP_TOL
    ]

    def agg(key):
        m, s, n = mean_std([per_seed[s][key] for s in seeds])
        return {"mean": m, "std": s, "n": n}

    result = {
        "name": args.name,
        "comment": args.comment,
        "git": git_info(),
        "config": args.config,
        "eval_split": split,
        "seeds": seeds,
        "walltime_min_per_seed": overrides.get("train.max_walltime_min"),
        "wall_total_min": round((time.time() - t0) / 60, 1),
        "params_M": params_m,
        "trt_export_flagged": trt_flagged,
        "per_seed": per_seed,
        "agg": {k: agg(k) for k in ("mAP_50_95", "f1", "lat_torch", "lat_trt")},
    }
    out = run_root / "candidate_result.json"
    out.write_text(json.dumps(result, indent=2))
    a = result["agg"]
    print(f"\n✅ wrote {out}")
    print(
        f"   {split}(mean/{len(seeds)} seeds): mAP_50_95={a['mAP_50_95']['mean']} "
        f"(±{a['mAP_50_95']['std']})  f1={a['f1']['mean']} (±{a['f1']['std']})"
    )
    print(
        f"   latency_ms: torch={a['lat_torch']['mean']} trt={a['lat_trt']['mean']}  params_M={params_m}"
    )
    if trt_flagged:
        print(
            f"   ⚠️  TRT export FLAGGED on seeds {trt_flagged}: torch vs TRT f1 differ by > "
            f"{TRT_F1_GAP_TOL}. Investigate the engine before trusting the verdict (EXPERIMENT_GUIDE §3)."
        )


if __name__ == "__main__":
    sys.exit(main())
