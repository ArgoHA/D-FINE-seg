#!/usr/bin/env python
"""Cityscapes regression test for D-FINE-seg: train -> TRT export -> bench, per task, vs the
reference in `regression_reference.json` (`--bless --bless-sources TASK=PATH` rewrites it). Needs a GPU box:
gitignored `configs/` presets, Cityscapes under `train.root`, TensorRT GPU.

Compared per task (margins per check, abs; latency is a ratio):

    check                      metrics                                      margin
    bench(torch) >= train      det/seg f1 iou | sem mIoU pxacc               0.01  --tol-bench
    |bench(trt) - bench(torch)| same                                         0.01  --tol-trt
    best-over-shared-epochs    det f1 mAP_50 iou | seg f1 mAP_50_mask iou    0.05  --tol-traj
      vs the reference        | sem mIoU pixel_acc
    TRT latency vs reference   ms                                            1.05x --tol-lat
    bench(trt) vs reference    det/seg f1 | sem mIoU (the prod number)       0.02  --tol-prod

Bench sits slightly above train (training eval has no NMS) - below means the export/infer
path broke. NMS/postprocess changes show up only in the bench(trt)-vs-reference check, so
the epoch trajectory can look fine while the number prod serves silently moved. PyTorch latency is INFO-only (host load). seg latency includes mask postprocess,
so an under-trained seg model reads *faster*.

Wall clock on a 5070 Ti (train + TRT export + bench):

    det   1.4 h + 5.4 m + 2.5 m  ~=  1.6 h   (55 epochs)
    sem   1.7 h + 5.3 m + 2.2 m  ~=  1.9 h   (75 epochs)
    seg   8.5 h + 6 m   + 20 m   ~=  9.0 h   (55 epochs)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
REFERENCE = Path(__file__).resolve().parent / "regression_reference.json"

# Below this a metric is still at its untrained floor, so comparing it to anything is
# vacuous - such rows are reported as WARN, never as a pass.
DEAD_METRIC = 0.05

# Below this many completed epochs the run is a smoke test, not a regression test: the model
# is still degenerate, so every comparison is indicative only and nothing may report FAIL.
# (Measured: a 1-epoch seg model benched TRT 0.054 f1 below torch purely because its scores
# sit on the conf threshold, where fp16 differences flip detections wholesale.)
MIN_EPOCHS = 3

# Metrics compared per task: first is the headline, the rest are corroborating.
TASKS = {
    "det": {
        "config": "config_cityscapes_det",
        "metrics": ["f1", "mAP_50", "iou"],
        "bench_metrics": ["f1", "iou"],  # bench runs at one conf thresh, so no mAPs
        "headline": "f1",  # the one number the notification carries
    },
    "seg": {
        "config": "config_cityscapes_seg",
        "metrics": ["f1", "mAP_50_mask", "iou"],
        "bench_metrics": ["f1", "iou"],
        "headline": "f1",
    },
    "sem": {
        "config": "config_city_sem",
        "metrics": ["mIoU", "pixel_acc"],
        "bench_metrics": ["mIoU", "pixel_acc"],
        "headline": "mIoU",
    },
}

# Blessing sources are passed per call (`--bless-sources TASK=PATH`) so no machine paths
# live in the repo. Each must be a full-length run that was also exported + benched, so
# latency comes from it.


# ---------------------------------------------------------------- parsing


def _tables(log: str):
    """Yield (epoch, {row: {metric: value}}) for every tabulate block in a train log.

    epoch is the int from "Metrics on epoch N", or 0 for the "Best epoch metrics" block.
    """
    header = re.compile(r"Metrics on epoch (\d+):|Best epoch metrics:")
    lines = log.splitlines()
    for i, line in enumerate(lines):
        m = header.search(line)
        if not m:
            continue
        epoch = int(m.group(1)) if m.group(1) else 0
        cells = [
            [c.strip() for c in row.strip().strip("|").split("|")]
            for row in lines[i + 1 : i + 12]
            if row.startswith("|")
        ]
        if not cells:
            continue
        cols, rows = cells[0][1:], {}
        for row in cells[1:]:
            vals = {}
            for k, v in zip(cols, row[1:]):
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
            rows[row[0]] = vals
        yield epoch, rows


def parse_train_log(run_dir: Path) -> dict:
    log = (run_dir / "train_log.txt").read_text()
    epochs, best = {}, {}
    for epoch, rows in _tables(log):
        if "val" not in rows:
            continue
        (best if epoch == 0 else epochs.setdefault(str(epoch), {})).update(rows["val"])
    out = {"epochs": epochs, "best": best}
    if m := re.search(r"Optimal batch size: (\d+)", log):
        out["batch_size"] = int(m.group(1))
    if m := re.search(r"Images in train: (\d+), val: (\d+)", log):
        out["split"] = [int(m.group(1)), int(m.group(2))]
    return out


def parse_bench(run_dir: Path) -> dict:
    """bench_metrics.csv -> {backend: {metric: value}}."""
    path = run_dir / "bench_metrics.csv"
    if not path.is_file():
        return {}
    rows = [r.split(",") for r in path.read_text().strip().splitlines()]
    cols = rows[0][1:]
    out = {}
    for row in rows[1:]:
        out[row[0]] = {k: float(v) for k, v in zip(cols, row[1:])}
    return out


def run_dir_for(config: str, exp_name: str) -> Path:
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now().strftime(pattern), replace=True
    )
    cfg = OmegaConf.load(CONFIGS / f"{config}.yaml")
    cfg.exp_name = exp_name
    path = Path(OmegaConf.to_container(cfg, resolve=True)["train"]["path_to_save"])
    if path.is_dir():
        return path
    # `exp` is stamped with the date at train start, so a run that crosses midnight (a full
    # seg run is ~9 h) leaves its results in yesterday's dir. Fall back to the newest match.
    dated = sorted(path.parent.glob(f"{exp_name}_*"), key=lambda p: p.stat().st_mtime)
    return dated[-1] if dated else path


# ---------------------------------------------------------------- running


def _stage(module: str, config: str, overrides: list, timings: dict) -> bool:
    cmd = [sys.executable, "-m", module, "-cp", str(CONFIGS), "-cn", config, *overrides]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    timings[module.rsplit(".", 1)[-1]] = (time.time() - t0) / 60
    return rc == 0


def run_task(task: str, minutes, ref: dict, timings: dict) -> Path:
    spec = TASKS[task]
    config, exp_name = spec["config"], f"regr_{task}"
    common = [f"exp_name={exp_name}", "train.use_wandb=False"]
    if bs := ref.get("batch_size"):  # pin: auto batch size follows free VRAM
        common.append(f"train.batch_size={bs}")
    for k, v in (ref.get("pinned") or {}).items():
        common.append(f"train.{k}={v}")

    train_overrides = list(common)
    if minutes:
        train_overrides.append(f"train.max_walltime_min={minutes}")
    if not _stage("dfine_seg.dl.train", config, train_overrides, timings):
        # A non-zero exit is a real error (OOM, bad config) - a ctrl+c still exits 0 and
        # leaves a usable best model. Exporting a failed run would only confuse the report.
        print(f"[{task}] training failed - skipping export/bench", file=sys.stderr)
        return run_dir_for(config, exp_name)
    # Export/bench only tensorrt+torch: onnx/openvino/coreml add minutes and are not
    # what this test is about (OpenVINO sem_seg alone is ~1 s/img).
    _stage("dfine_seg.dl.export", config, common + ["export.formats=[tensorrt]"], timings)
    _stage("dfine_seg.dl.bench", config, common + ["bench.formats=[torch,tensorrt]"], timings)
    return run_dir_for(config, exp_name)


# ---------------------------------------------------------------- checking


def check(task: str, run_dir: Path, ref: dict, tol) -> list:
    """-> list of (level, text); level in {ok, warn, fail}."""
    spec, out = TASKS[task], []
    got = parse_train_log(run_dir)
    bench = parse_bench(run_dir)
    smoke = len(got["epochs"]) < MIN_EPOCHS

    if ref.get("split") and got.get("split") and ref["split"] != got["split"]:
        out.append(("fail", f"split {got['split']} != reference {ref['split']} (check symlinks)"))
    # Only meaningful when the run auto-picked; the harness normally pins it to the reference.
    if got.get("batch_size") and got["batch_size"] != ref.get("batch_size"):
        out.append(("warn", f"batch {got['batch_size']} != reference {ref.get('batch_size')}"))

    # 1. bench(PyTorch) >= train, on the best model
    torch_row = bench.get("PyTorch", {})
    for m in spec["bench_metrics"]:
        if m in torch_row and m in got["best"]:
            d = torch_row[m] - got["best"][m]
            # Comparing two ~0 metrics proves nothing - an under-trained run has to say so
            # rather than bank a free pass.
            if max(torch_row[m], got["best"][m]) < DEAD_METRIC:
                lvl = "warn"
            else:
                lvl = "ok" if d >= -tol.bench else "fail"
            out.append(
                (
                    lvl,
                    f"bench torch {m} {torch_row[m]:.4f} vs train {got['best'][m]:.4f} ({d:+.4f})",
                )
            )
    # 2. TensorRT ~= PyTorch
    trt_row = bench.get("TensorRT", {})
    for m in spec["bench_metrics"]:
        if m in trt_row and m in torch_row:
            d = trt_row[m] - torch_row[m]
            if max(trt_row[m], torch_row[m]) < DEAD_METRIC:
                lvl = "warn"
            else:
                lvl = "ok" if abs(d) <= tol.trt else "fail"
            out.append(
                (lvl, f"bench trt   {m} {trt_row[m]:.4f} vs torch {torch_row[m]:.4f} ({d:+.4f})")
            )
    if not bench:
        out.append(("fail", "no bench_metrics.csv - export or bench did not finish"))

    # 3. TensorRT latency must not degrade. Engine speed does not depend on convergence,
    # but seg's postprocess scales with detection count, so a short run reads artificially
    # fast - that direction never fails. PyTorch latency is host-load dependent: info only.
    ref_lat = ref.get("latency") or {}
    for backend in ("TensorRT", "PyTorch"):
        new_ms, ref_ms = bench.get(backend, {}).get("latency"), ref_lat.get(backend)
        if not new_ms or not ref_ms:
            continue
        ratio = new_ms / ref_ms
        lvl = "info" if backend == "PyTorch" else ("ok" if ratio <= tol.lat else "fail")
        out.append(
            (lvl, f"{backend:8s} latency {new_ms:.1f}ms vs ref {ref_ms:.1f}ms (x{ratio:.2f})")
        )

    # 3b. prod: the number that actually ships. Training eval has no NMS, so a NMS/postprocess
    # change never shows in the trajectory - only in bench(trt) vs the reference.
    ref_bench = (ref.get("bench") or {}).get("TensorRT", {})
    head = spec["headline"]
    prod_val, ref_val = trt_row.get(head), ref_bench.get(head)
    if prod_val is not None and ref_val is not None:
        d = prod_val - ref_val
        lvl = (
            "warn"
            if max(prod_val, ref_val) < DEAD_METRIC
            else ("ok" if abs(d) <= tol.prod else "fail")
        )
        out.append((lvl, f"prod trt   {head} {prod_val:.4f} vs ref {ref_val:.4f} ({d:+.4f})"))

    # 4. trajectory: best-so-far over the epochs both runs share. Comparing the *best* of a
    # window rather than one epoch mirrors what model.pt actually is, and does not hinge on
    # which epoch the walltime cap happened to stop at.
    shared = sorted(set(got["epochs"]) & set(ref.get("epochs", {})), key=int)
    if not shared:
        out.append(("warn", "no epoch overlaps the reference"))
        return out
    for m in spec["metrics"]:
        pairs = [
            (e, got["epochs"][e][m], ref["epochs"][e][m])
            for e in shared
            if m in got["epochs"][e] and m in ref["epochs"][e]
        ]
        if not pairs:
            continue
        best_got, best_ref = max(p[1] for p in pairs), max(p[2] for p in pairs)
        d = best_got - best_ref
        if max(best_got, best_ref) < DEAD_METRIC:
            lvl = "warn"
        else:
            lvl = "ok" if abs(d) <= tol.traj else "fail"
        worst_e, worst = max(((e, g - r) for e, g, r in pairs), key=lambda kv: abs(kv[1]))
        out.append(
            (
                lvl,
                f"ep1-{shared[-1]:<2} best {m} {best_got:.4f} vs ref {best_ref:.4f} ({d:+.4f});"
                f" worst single epoch {worst:+.4f} @ep{worst_e}",
            )
        )
    if smoke:
        # Nothing here is trustworthy yet: downgrade rather than cry regression.
        out = [("warn", t) if lvl == "fail" else (lvl, t) for lvl, t in out]
        out.insert(0, ("warn", f"only {len(got['epochs'])} epoch(s) trained - indicative only"))
    return out


# Reference updates are a deliberate, manual act (--bless) - a passing check never rewrites
# the reference. Only after a real change to the training/data/export path, and only from a
# full-length run that was also exported + benched and whose frozen config matches the preset.
# bless() does not verify those itself - that is why it is manual.


def bless(sources: dict) -> None:
    ref = {}
    for task, run in sources.items():
        run_dir = Path(run)
        if not run_dir.is_dir():
            print(f"skipping {task}: {run_dir} missing", file=sys.stderr)
            continue
        entry = parse_train_log(run_dir)
        # Memory-shaped knobs from the frozen config. Pinning them keeps the comparison
        # honest AND avoids the host OOM that killed a seg final-eval at the preset's
        # num_workers=12 / mask_batch_size=150 (31 GB box, ~6 GB of it worker shmem).
        frozen = yaml.safe_load((run_dir / "config.yaml").read_text())["train"]
        entry["pinned"] = {
            k: frozen[k] for k in ("num_workers", "mask_batch_size") if frozen.get(k) is not None
        }
        # A run that was itself pinned never printed "Optimal batch size", so the log has
        # nothing to parse - take it off the frozen config instead.
        if "batch_size" not in entry and int(frozen.get("batch_size", -1)) > 0:
            entry["batch_size"] = int(frozen["batch_size"])
        # Keep only the compared metrics - the reference is committed, so keep it small.
        keep = set(TASKS[task]["metrics"]) | set(TASKS[task]["bench_metrics"])
        entry["epochs"] = {
            e: {k: v for k, v in vals.items() if k in keep} for e, vals in entry["epochs"].items()
        }
        entry["best"] = {k: v for k, v in entry["best"].items() if k in keep}
        entry["run"] = run_dir.name
        entry["config"] = TASKS[task]["config"]
        bench = parse_bench(run_dir)
        head = TASKS[task]["headline"]
        entry["bench"] = {
            b: {head: v[head]}
            for b, v in bench.items()
            if head in v and b in ("PyTorch", "TensorRT")
        }
        lat = {
            k: v["latency"]
            for k, v in bench.items()
            if "latency" in v and k in ("PyTorch", "TensorRT")
        }
        if lat:
            entry["latency"] = lat
        else:
            print(f"  warning: {run_dir.name} has no bench_metrics.csv - no latency baseline")
        ref[task] = entry
        print(f"{task}: {len(entry['epochs'])} epochs from {run_dir.name}")
    REFERENCE.write_text(json.dumps(ref, indent=1, sort_keys=True) + "\n")
    print(f"wrote {REFERENCE}")


def notification(results: dict, run_dirs: dict, failed: bool) -> str:
    """Two numbers per task - benched TensorRT headline metric and its latency - plus a
    verdict. Everything else lives in the console report; a phone screen gets the gist.
    """
    lines = [
        f"{'❌' if failed else '✅'} D-FINE-seg regression {'FAILED' if failed else 'PASSED'}",
        "",
    ]
    for task, rows in results.items():
        trt = parse_bench(run_dirs[task]).get("TensorRT", {})
        metric = TASKS[task]["headline"]
        value = trt.get(metric)
        latency = trt.get("latency")
        bad = any(lvl == "fail" for lvl, _ in rows)
        cells = [
            f"{metric} {value:.3f}" if value is not None else f"{metric} n/a",
            f"{latency:.1f} ms" if latency is not None else "n/a",
        ]
        lines.append(f"{'❌' if bad else '✅'} {task:<4} {cells[0]:<11} {cells[1]}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, help="walltime cap per training task")
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    ap.add_argument(
        "--tol-bench", type=float, default=0.01, help="how far bench may sit below train"
    )
    ap.add_argument("--tol-trt", type=float, default=0.01, help="max |TensorRT - PyTorch|")
    ap.add_argument("--tol-traj", type=float, default=0.05, help="max drift vs the reference epoch")
    ap.add_argument("--tol-lat", type=float, default=1.05, help="max TensorRT latency ratio vs ref")
    ap.add_argument(
        "--tol-prod", type=float, default=0.02, help="max |bench trt headline - reference|"
    )
    ap.add_argument("--check-only", action="store_true", help="re-check existing run dirs")
    ap.add_argument("--bless", action="store_true", help="rewrite the reference")
    ap.add_argument(
        "--bless-sources",
        nargs="+",
        metavar="TASK=PATH",
        default=None,
        help="run dirs to bless from (required by --bless), e.g. det=/path/to/run",
    )
    ap.add_argument("--notify", action="store_true", help="Telegram the summary via notify.py")
    args = ap.parse_args()

    if args.bless:
        if not args.bless_sources:
            ap.error("--bless requires --bless-sources TASK=PATH [TASK=PATH ...]")
        bless(dict(p.split("=", 1) for p in args.bless_sources))
        return 0

    reference = json.loads(REFERENCE.read_text()) if REFERENCE.is_file() else {}
    tol = argparse.Namespace(
        bench=args.tol_bench,
        trt=args.tol_trt,
        traj=args.tol_traj,
        lat=args.tol_lat,
        prod=args.tol_prod,
    )
    results, timings, run_dirs = {}, {}, {}
    for task in args.tasks:
        ref = reference.get(task, {})
        timings[task] = {}
        if args.check_only:
            run_dir = run_dir_for(TASKS[task]["config"], f"regr_{task}")
        else:
            run_dir = run_task(task, args.minutes, ref, timings[task])
        run_dirs[task] = run_dir
        if not (run_dir / "train_log.txt").is_file():
            results[task] = [("fail", f"no train log at {run_dir}")]
            continue
        results[task] = check(task, run_dir, ref, tol)

    mark = {"ok": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}
    failed = any(lvl == "fail" for rows in results.values() for lvl, _ in rows)
    header = f"REGRESSION {'FAILED' if failed else 'OK'}"
    report = [header + (f" ({args.minutes} min/task)" if args.minutes else " (full runs)")]
    for task, rows in results.items():
        stages = timings.get(task) or {}
        clock = "  ".join(f"{k} {v:.1f}m" for k, v in stages.items())
        report.append(f"\n{task}{'  (' + clock + ')' if clock else ''}")
        report += [f"  [{mark[lvl]}] {text}" for lvl, text in rows]
    summary = "\n".join(report)
    print(f"\n{'=' * 72}\n{summary}\n")

    if args.notify:  # notify.py only warns when TG creds are absent
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from notify import send

            send(notification(results, run_dirs, failed))
        except Exception as e:
            print(f"notification skipped: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
