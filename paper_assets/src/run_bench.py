#!/usr/bin/env python3
"""
End-to-end benchmarking pipeline for D-FINE-seg.

Three phases, each over the 5 model sizes (n, s, m, l, x):
  1. VisDrone — object detection
  2. TACO     — object detection      (swaps labels_det <-> labels)
  3. TACO     — instance segmentation (swaps labels_seg <-> labels)

For each phase:
  - copy configs/<config>.yaml -> config.yaml
  - (TACO) make sure dataset/labels/ holds the right label set for the task
  - export all 5 sizes in parallel (PARALLEL_EXPORTS at a time)
  - bench all 5 sizes sequentially (only after every export is done)
  - read bench_metrics.csv from each exp dir

Final step writes bench_results.md with three F1 + latency tables.

Hardcoded knobs (edit at the top of this file):
  TO_EXPORT        — True by default; set False to skip export and only re-bench
  PARALLEL_EXPORTS — number of concurrent exports per phase (default 2)

Usage:
  python run_bench.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ── Knobs ─────────────────────────────────────────────────────────────────────
TO_EXPORT = False
PARALLEL_EXPORTS = 3

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"
CONFIG_YAML = REPO_ROOT / "config.yaml"
LOG_DIR = REPO_ROOT / "logs" / "bench_pipeline"
RESULTS_MD = REPO_ROOT / "bench_results.md"
MODEL_SIZES = ("n", "s", "m", "l", "x")


PHASES = (
    {
        "name": "VisDrone — Object Detection",
        "config": "config_drone.yaml",
        "task": "detect",
        "exp_prefix": "det",
        "labels_target": None,
    },
    {
        "name": "TACO — Object Detection",
        "config": "config_taco.yaml",
        "task": "detect",
        "exp_prefix": "det",
        "labels_target": "labels_det",
    },
    {
        "name": "TACO — Instance Segmentation",
        "config": "config_taco.yaml",
        "task": "segment",
        "exp_prefix": "seg",
        "labels_target": "labels_seg",
    },
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_train_root(config_path: Path) -> Path:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["train"]["root"])


def ensure_labels_active(data_path: Path, target: str) -> None:
    """Make sure data_path/labels holds the `target` label set.

    `target` is "labels_det" or "labels_seg". The other set lives in the
    sibling directory of the same name when not active. Idempotent: if labels/
    already holds the target (no sibling target dir present), do nothing.
    """
    if target not in {"labels_det", "labels_seg"}:
        raise ValueError(f"unexpected target: {target!r}")
    other = "labels_seg" if target == "labels_det" else "labels_det"

    labels = data_path / "labels"
    target_dir = data_path / target
    other_dir = data_path / other

    if target_dir.exists():
        if labels.exists() and other_dir.exists():
            raise RuntimeError(
                f"ambiguous state in {data_path}: labels/, {target}/, and {other}/ all exist"
            )
        if labels.exists():
            labels.rename(other_dir)
        target_dir.rename(labels)
        log(f"  swapped labels: {target} -> labels (preserved old as {other})")
    elif labels.exists():
        log(f"  labels/ assumed to already hold {target} (no sibling {target}/ found)")
    else:
        raise RuntimeError(f"neither labels/ nor {target}/ exists in {data_path}")


def find_exp_dir(prefix: str, size: str, train_root: Path) -> Optional[Path]:
    """Pick the most recent <prefix>_<size>_<date> dir under <root>/output/models."""
    base = train_root / "output" / "models"
    if not base.exists():
        return None
    candidates = [
        d for d in base.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}_{size}_")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.name.rsplit("_", 1)[-1])
    return candidates[-1]


def run_subprocess(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def export_one(phase: dict, size: str) -> tuple[str, int, Path]:
    exp_name = f"{phase['exp_prefix']}_{size}"
    log_path = LOG_DIR / f"export__{exp_name}.log"
    cmd = [
        sys.executable,
        "-m",
        "src.dl.export",
        f"exp_name={exp_name}",
        f"model_name={size}",
        f"task={phase['task']}",
    ]
    rc = run_subprocess(cmd, log_path)
    return exp_name, rc, log_path


def bench_one(phase: dict, size: str) -> tuple[str, int, Path]:
    exp_name = f"{phase['exp_prefix']}_{size}"
    log_path = LOG_DIR / f"bench__{exp_name}.log"
    cmd = [
        sys.executable,
        "-m",
        "src.dl.bench",
        f"exp_name={exp_name}",
        f"model_name={size}",
        f"task={phase['task']}",
    ]
    rc = run_subprocess(cmd, log_path)
    return exp_name, rc, log_path


def read_bench_metrics(exp_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = exp_dir / "bench_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, index_col=0)


def write_results(all_results: dict[str, dict[str, pd.DataFrame]]) -> None:
    """all_results: {phase_name: {size: bench_metrics_df}} where df is indexed by backend."""
    lines = ["# D-FINE-seg benchmark results", ""]
    lines.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `run_bench.py`.")
    lines.append("")

    for phase in PHASES:
        lines.append(f"## {phase['name']}")
        lines.append("")
        size_to_df = all_results.get(phase["name"], {})

        backends: list[str] = []
        for df in size_to_df.values():
            if df is None:
                continue
            for fmt in df.index:
                if fmt not in backends:
                    backends.append(fmt)

        if not backends:
            lines.append("_no results_")
            lines.append("")
            continue

        for backend in backends:
            lines.append(f"### Backend: {backend}")
            lines.append("")
            lines.append("| Size | F1-score | IoU | Precision | Recall | Latency (ms) |")
            lines.append("|:----:|:--------:|:---:|:---------:|:------:|:------------:|")
            for size in MODEL_SIZES:
                df = size_to_df.get(size)
                if df is None or backend not in df.index:
                    lines.append(f"| {size.upper()} | — | — | — | — | — |")
                    continue
                row = df.loc[backend]
                f1 = row.get("f1", "—")
                iou = row.get("iou", "—")
                prec = row.get("precision", "—")
                rec = row.get("recall", "—")
                lat = row.get("latency", "—")
                lines.append(f"| {size.upper()} | {f1} | {iou} | {prec} | {rec} | {lat} |")
            lines.append("")

    RESULTS_MD.write_text("\n".join(lines))
    log(f"wrote {RESULTS_MD.relative_to(REPO_ROOT)}")


def run_phase(phase: dict) -> dict[str, pd.DataFrame]:
    log(f"=== {phase['name']} ===")

    src_cfg = CONFIG_DIR / phase["config"]
    shutil.copy2(src_cfg, CONFIG_YAML)
    log(f"  copied configs/{src_cfg.name} -> config.yaml")

    train_root = load_train_root(CONFIG_YAML)
    log(f"  train.root = {train_root}")

    if phase["labels_target"]:
        ensure_labels_active(train_root / "data" / "dataset", phase["labels_target"])

    if TO_EXPORT:
        log(f"  exporting {len(MODEL_SIZES)} sizes (max {PARALLEL_EXPORTS} in parallel)")
        with ThreadPoolExecutor(max_workers=PARALLEL_EXPORTS) as ex:
            futures = {ex.submit(export_one, phase, s): s for s in MODEL_SIZES}
            for fut in as_completed(futures):
                exp_name, rc, log_path = fut.result()
                status = (
                    "ok" if rc == 0 else f"FAIL (rc={rc}, see {log_path.relative_to(REPO_ROOT)})"
                )
                log(f"    export {exp_name}: {status}")
    else:
        log("  skipping exports (TO_EXPORT=False)")

    log("  benchmarking (sequential)")
    results: dict[str, pd.DataFrame] = {}
    for size in MODEL_SIZES:
        exp_name, rc, log_path = bench_one(phase, size)
        if rc != 0:
            log(f"    bench {exp_name}: FAIL (rc={rc}, see {log_path.relative_to(REPO_ROOT)})")
            continue
        exp_dir = find_exp_dir(phase["exp_prefix"], size, train_root)
        if exp_dir is None:
            log(f"    bench {exp_name}: ok, but no exp dir found")
            continue
        df = read_bench_metrics(exp_dir)
        if df is None:
            log(f"    bench {exp_name}: ok, but no bench_metrics.csv in {exp_dir.name}")
            continue
        results[size] = df
        first = df.iloc[0]
        log(
            f"    bench {exp_name}: ok ("
            f"f1={first.get('f1', '?')}, "
            f"iou={first.get('iou', '?')}, "
            f"p={first.get('precision', '?')}, "
            f"r={first.get('recall', '?')}, "
            f"latency={first.get('latency', '?')})"
        )

    return results


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    backup = CONFIG_YAML.with_suffix(".yaml.run_bench_bak")
    if CONFIG_YAML.exists():
        shutil.copy2(CONFIG_YAML, backup)
        log(f"backed up config.yaml -> {backup.name}")

    all_results: dict[str, dict[str, pd.DataFrame]] = {}
    try:
        for phase in PHASES:
            all_results[phase["name"]] = run_phase(phase)
    finally:
        if backup.exists():
            shutil.copy2(backup, CONFIG_YAML)
            backup.unlink()
            log("restored config.yaml")
        write_results(all_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
