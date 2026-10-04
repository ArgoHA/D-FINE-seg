"""Send a Telegram message using TG_TOKEN / TG_CHAT_ID from the env or .env. Missing creds only warn.

uv run python experiments/notify.py "✅ lovasz: mIoU 0.702 (+0.006) @ 2.01 ms. Next: ..."
"""

import os
import sys
import time
from pathlib import Path

import requests

ENV = Path(__file__).resolve().parents[1] / ".env"


def creds() -> tuple[str | None, str | None]:
    env = dict(os.environ)
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            k, _, v = line.removeprefix("export ").partition("=")
            env.setdefault(k.strip(), v.strip().strip("'\""))
    return env.get("TG_TOKEN"), env.get("TG_CHAT_ID")


def send(message: str):
    tok, chat = creds()
    if not tok or not chat:
        print("TG_TOKEN/TG_CHAT_ID not set, skipping notification")
        return
    for attempt in range(5):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                json={"chat_id": chat, "text": message},
                timeout=20,
            )
            print("sent" if r.ok else f"send failed ({r.status_code})")
            return
        except requests.RequestException as e:  # never print e: its URL contains the token
            print(f"send attempt {attempt + 1} failed: {type(e).__name__}")
            time.sleep(30)


def main():
    send(" ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
