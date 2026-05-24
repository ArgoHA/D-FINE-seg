"""Quick-look a multi-channel .npy image on macOS.

Splits the array into:
  - <stem>_rgb.jpg     first 3 channels (assumed RGB by repo convention)
  - <stem>_ch<N>.jpg   each extra channel rendered with a thermal colormap

Drops the JPEGs in /tmp and opens them in Preview.app.

Usage:
    uv run python scripts/preview_npy.py /path/to/img.npy [more.npy ...]

Wire it as a Finder Quick Action:
cp scripts/preview_npy.py ~/bin/preview_npy.py

quick action -> run shell script -> /bin/bash absolute/path/to/python /Users/username/bin/preview_npy.py "$@"
pass input as argument
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

TMP = Path(tempfile.gettempdir()) / "npy_preview"
TMP.mkdir(exist_ok=True)


def render(path: Path) -> list[Path]:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.dtype != np.uint8:
        # Normalize floats / uint16 to uint8 for display only.
        a = arr.astype(np.float32)
        a -= a.min()
        if a.max() > 0:
            a *= 255.0 / a.max()
        arr = a.astype(np.uint8)

    outs: list[Path] = []
    stem = path.stem
    H, W, C = arr.shape

    if C >= 3:
        # First 3 channels = RGB; cv2 writes BGR, so swap.
        rgb = arr[..., :3][..., ::-1]
        p = TMP / f"{stem}_rgb.jpg"
        cv2.imwrite(str(p), rgb)
        outs.append(p)

    extras = range(3, C) if C >= 3 else range(C)
    for ch in extras:
        colored = cv2.applyColorMap(arr[..., ch], cv2.COLORMAP_INFERNO)
        p = TMP / f"{stem}_ch{ch}.jpg"
        cv2.imwrite(str(p), colored)
        outs.append(p)

    print(f"{path}  shape={arr.shape} dtype={arr.dtype}  ->  {len(outs)} preview(s)")
    return outs


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip())
        return 1

    all_outputs: list[Path] = []
    for raw in argv:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"skip: not found  {path}", file=sys.stderr)
            continue
        if path.suffix.lower() != ".npy":
            print(f"skip: not .npy  {path}", file=sys.stderr)
            continue
        try:
            all_outputs.extend(render(path))
        except Exception as exc:
            print(f"skip: {path}: {exc}", file=sys.stderr)

    if all_outputs:
        subprocess.run(["open", "-a", "Preview", *map(str, all_outputs)], check=False)
    return 0 if all_outputs else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
