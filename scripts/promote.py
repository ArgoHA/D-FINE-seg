#!/usr/bin/env python
"""
Decide whether a candidate beats the current best, enforce the frozen-eval guard,
keep the ledger, and (on the first run) establish the persistent baseline. Pure
decision + bookkeeping — it does NOT move git branches (that stays agent-driven per
EXPERIMENT_GUIDE.md) so nothing silently rewrites your repo.

`baseline.json` always describes the CURRENT BEST (not the original control). It is
established once on the first run and then overwritten whenever a candidate is
promoted, so a fresh agent never re-trains a baseline — it just reads this file.

Two metrics, both on the held-out TEST set, mean over seeds:
  - mAP_50_95  (from training)  — primary; must improve beyond its noise margin.
  - f1         (from bench)     — guard;   must not regress beyond its noise margin.
`margin` = the current best's across-seed std for that metric (floor 0.003), i.e. a
move must clear the variance we actually measured.

    gain_map = cand_map - best_map ;  gain_f1 = cand_f1 - best_f1
    lat_ratio = cand_latency / best_latency   (TensorRT, fallback PyTorch)
    PROMOTE if  gain_f1 > -f1_margin  (no real f1 regression)  AND
                ( (gain_map > map_margin   and lat_ratio <= 1.05)
                  or (gain_map > 2*map_margin and lat_ratio <= 1.20) )

Usage:
    # first run establishes the baseline automatically:
    uv run python scripts/promote.py --candidate experiments/runs/baseline/candidate_result.json
    # later candidates are judged against the current best:
    uv run python scripts/promote.py --candidate experiments/runs/<name>/candidate_result.json --base main_exp
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "experiments" / "baseline.json"
LEDGER = REPO / "experiments" / "ledger.csv"

# Files the research loop must NOT change — they define how we measure success.
# A candidate diff touching these is rejected regardless of metrics.
FROZEN = ["src/dl/validator.py", "src/dl/bench.py", "scripts/run_candidate.py", "scripts/promote.py"]

LAT_TIGHT, LAT_LOOSE, MARGIN_FLOOR = 1.05, 1.20, 0.003
LEDGER_COLS = [
    "timestamp", "name", "branch", "sha", "base", "seeds", "split",
    "map_mean", "map_gain", "map_margin", "f1_mean", "f1_gain", "f1_margin",
    "lat_torch_ms", "lat_trt_ms", "lat_ratio", "params_M", "promoted", "comment",
]


def latency(res):
    """(trt_mean, torch_mean) from a candidate_result.json or baseline.json shape."""
    if "agg" in res:  # candidate
        return res["agg"]["lat_trt"]["mean"], res["agg"]["lat_torch"]["mean"]
    lat = res.get("latency_ms", {})  # baseline
    return lat.get("trt"), lat.get("torch")


def frozen_violations(base):
    try:
        committed = subprocess.run(["git", "diff", "--name-only", base], cwd=REPO,
                                   capture_output=True, text=True)
        uncommitted = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                     capture_output=True, text=True)
    except Exception as e:  # noqa
        print(f"⚠️  could not run git for frozen-path check: {e}")
        return None
    changed = set(committed.stdout.split())
    changed |= {ln[3:].strip() for ln in uncommitted.stdout.splitlines() if ln.strip()}
    return sorted(f for f in FROZEN if f in changed)


def baseline_from(cand):
    a = cand["agg"]
    trt, torch_l = latency(cand)
    return {
        "name": cand["name"], "eval_split": cand.get("eval_split", "test"), "git": cand["git"],
        "means": {"mAP_50_95": a["mAP_50_95"]["mean"], "f1": a["f1"]["mean"]},
        "margin": {
            "mAP_50_95": round(max(a["mAP_50_95"]["std"], MARGIN_FLOOR), 4),
            "f1": round(max(a["f1"]["std"], MARGIN_FLOOR), 4),
        },
        "latency_ms": {"trt": trt, "torch": torch_l}, "params_M": cand["params_M"],
    }


def append_ledger(row):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--base", default="main_exp", help="git ref the candidate branched from (frozen-path check)")
    args = ap.parse_args()

    cand = json.loads(Path(args.candidate).read_text())
    a = cand["agg"]
    split = cand.get("eval_split", "test")
    cand_map, cand_f1 = a["mAP_50_95"]["mean"], a["f1"]["mean"]
    cand_trt, cand_torch = latency(cand)

    # --- first run ever: establish the persistent baseline (the control) ---
    if not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(baseline_from(cand), indent=2))
        append_ledger({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": cand["name"], "branch": cand["git"]["branch"], "sha": cand["git"]["sha"],
            "base": args.base, "seeds": len(cand["seeds"]), "split": split,
            "map_mean": cand_map, "map_gain": "", "map_margin": "",
            "f1_mean": cand_f1, "f1_gain": "", "f1_margin": "",
            "lat_torch_ms": cand_torch, "lat_trt_ms": cand_trt, "lat_ratio": "",
            "params_M": cand["params_M"], "promoted": True,
            "comment": cand.get("comment", "") + " [baseline/control]",
        })
        print(f"✅ baseline established from '{cand['name']}' ({split}): "
              f"mAP_50_95={cand_map}  f1={cand_f1}\n   margins: {baseline_from(cand)['margin']}")
        print("   Commit experiments/baseline.json so future agents reuse it (never re-train the control).")
        return 0

    base = json.loads(BASELINE.read_text())
    base_map, base_f1 = base["means"]["mAP_50_95"], base["means"]["f1"]
    m_map = base["margin"].get("mAP_50_95", MARGIN_FLOOR)
    m_f1 = base["margin"].get("f1", MARGIN_FLOOR)
    base_trt, base_torch = base["latency_ms"].get("trt"), base["latency_ms"].get("torch")

    gain_map = round(cand_map - base_map, 4)
    gain_f1 = round(cand_f1 - base_f1, 4)
    cand_lat, base_lat = (cand_trt, base_trt) if cand_trt and base_trt else (cand_torch, base_torch)
    lat_ratio = round(cand_lat / base_lat, 3) if cand_lat and base_lat else None

    viol = frozen_violations(args.base)
    blocked = bool(viol)

    f1_ok = gain_f1 > -m_f1
    within_tight = lat_ratio is not None and lat_ratio <= LAT_TIGHT
    within_loose = lat_ratio is not None and lat_ratio <= LAT_LOOSE
    promote = (not blocked) and f1_ok and (
        (gain_map > m_map and within_tight) or (gain_map > 2 * m_map and within_loose)
    )

    print(f"\n=== {cand['name']} vs current best '{base['name']}' [{split}] ===")
    print(f" mAP_50_95 : {cand_map}  (best {base_map}, gain {gain_map:+}, margin {m_map}, 2x {2*m_map:.4f})  PRIMARY")
    print(f" f1        : {cand_f1}  (best {base_f1}, gain {gain_f1:+}, margin {m_f1})  GUARD"
          f"{'' if f1_ok else '  ❌ regressed'}")
    print(f" latency_ms: cand {cand_lat} / best {base_lat}  ratio {lat_ratio} (tight {LAT_TIGHT}, loose {LAT_LOOSE})")
    print(f" params_M  : cand {cand['params_M']} / best {base['params_M']}")
    if blocked:
        print(f" ❌ FROZEN-PATH VIOLATION (rejected): {viol}")
    print(f"\n VERDICT: {'🟢 PROMOTE' if promote else '🔴 KEEP CURRENT BEST'}")
    print(" Reminder: if the gain is marginal but the change adds real complexity, prefer to KEEP "
          "(simplicity rule) — your call, note it in the lab notebook.")
    if promote:
        BASELINE.write_text(json.dumps(baseline_from(cand), indent=2))
        print(" ↑ baseline.json updated to this candidate (new current best).")
        print(" Next (agent, per EXPERIMENT_GUIDE.md): git branch -f main_exp HEAD ; commit ledger+notebook+baseline")

    append_ledger({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": cand["name"], "branch": cand["git"]["branch"], "sha": cand["git"]["sha"],
        "base": args.base, "seeds": len(cand["seeds"]), "split": split,
        "map_mean": cand_map, "map_gain": gain_map, "map_margin": m_map,
        "f1_mean": cand_f1, "f1_gain": gain_f1, "f1_margin": m_f1,
        "lat_torch_ms": cand_torch, "lat_trt_ms": cand_trt, "lat_ratio": lat_ratio,
        "params_M": cand["params_M"], "promoted": promote, "comment": cand.get("comment", ""),
    })
    print(f" 📒 appended to {LEDGER.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
