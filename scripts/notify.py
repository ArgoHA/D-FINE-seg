"""Telegram notifier for finished experiments.

Sends a short summary to the chat in .env (TG_TOKEN / TG_CHAT_ID). Use after
promote.py so the ledger holds the verdict:

    uv run python scripts/notify.py --name <name>     # summarize a candidate
    uv run python scripts/notify.py --message "text"  # arbitrary note (e.g. a failure)

Missing creds only warn (notification is auxiliary — never break the run).
"""

import argparse
import csv
import json
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "experiments" / "ledger.csv"
ENV = REPO / ".env"


def load_creds() -> tuple[str | None, str | None]:
    import os

    tok, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT_ID")
    if (not tok or not chat) and ENV.exists():
        for line in ENV.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip("'\"")
            if k.strip() == "TG_TOKEN":
                tok = tok or v
            elif k.strip() == "TG_CHAT_ID":
                chat = chat or v
    return tok, chat


def last_ledger_row(name: str) -> dict | None:
    if not LEDGER.exists():
        return None
    rows = [r for r in csv.DictReader(LEDGER.open()) if r["name"] == name]
    return rows[-1] if rows else None


def summarize(name: str) -> str:
    res = json.loads((REPO / "experiments" / "runs" / name / "candidate_result.json").read_text())
    a = res["agg"]
    row = last_ledger_row(name) or {}
    promoted = str(row.get("promoted", "")).lower() == "true"
    has_gain = bool(row.get("map_gain", ""))  # baseline row leaves gains empty

    if not row:
        head = f"✅ finished — {name}"
    elif has_gain:
        head = f"{'🟢 PROMOTE' if promoted else '🔴 KEEP BEST'} — {name}"
    else:
        head = f"⭐ baseline established — {name}"

    def gain(k):
        g = row.get(k, "")
        return f"  (gain {float(g):+.4f})" if g not in ("", None) else ""

    lines = [
        head,
        res.get("comment", ""),
        f"mAP_50_95: {a['mAP_50_95']['mean']:.4f} ±{a['mAP_50_95']['std']:.4f}{gain('map_gain')}",
        f"f1:        {a['f1']['mean']:.4f} ±{a['f1']['std']:.4f}{gain('f1_gain')}",
        f"latency:   trt {a['lat_trt']['mean']:.1f}ms  torch {a['lat_torch']['mean']:.1f}ms"
        + (f"  (ratio {row['lat_ratio']})" if row.get("lat_ratio") else ""),
        f"params {res['params_M']:.2f}M · {len(res['seeds'])} seeds · {res['wall_total_min']:.0f}min wall",
        f"{res['git']['sha']} @ {res['git']['branch']}",
    ]
    return "\n".join(line for line in lines if line)


def send(text: str) -> bool:
    tok, chat = load_creds()
    if not tok or not chat:
        print("⚠️  TG_TOKEN/TG_CHAT_ID not set — skipping Telegram notification.")
        return False
    r = requests.post(
        f"https://api.telegram.org/bot{tok}/sendMessage",
        json={"chat_id": chat, "text": text},
        timeout=20,
    )
    if not r.ok:
        print(f"⚠️  Telegram send failed ({r.status_code}): {r.text[:200]}")
        return False
    print("✅ Telegram notification sent.")
    return True


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="candidate slug under experiments/runs/")
    g.add_argument("--message", help="send arbitrary text instead of a result summary")
    args = ap.parse_args()
    send(args.message if args.message else summarize(args.name))


if __name__ == "__main__":
    main()
