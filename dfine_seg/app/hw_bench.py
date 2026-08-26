"""`dfine hw_bench` - device throughput benchmark for the Torch inference path.

Config-free, like `predict`: one fixed synthetic 1920x1080 frame goes through the full
wrapper call (preprocess -> forward -> postprocess, exactly what `dfine predict` runs)
for a fixed wall-clock window, counting iterations. `--batch N` stacks N copies of the
frame per call - the standard way to saturate a GPU that a single batch-1 stream leaves
idle.

Device follows the wrapper's own auto chain (cuda -> mps -> cpu) unless `--device`
overrides it. Results go to stdout only; nothing is written to disk.
"""

import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

HEIGHT, WIDTH = 1080, 1920
SEED = 0
CONF_THRESH = 0.5
_SIZES = ("n", "s", "m", "l", "x")


def make_input(seed: int = SEED) -> np.ndarray:
    """Seeded 1920x1080 uint8 noise: content-independent, byte-identical on every host."""
    return np.random.default_rng(seed).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def run_bench(
    call: Callable[[Any], Any],
    imgs: Any,
    *,
    duration: float,
    warmup: int,
    sync: Callable[[], None],
) -> Dict[str, Any]:
    """Call `call(imgs)` until the wall-clock window closes, syncing the device per call.

    Returns {"calls", "elapsed", "ms"}: per-call wall times (ms), warmup not included.
    `sync` is `torch.cuda/mps.synchronize` for GPU devices, a no-op on CPU - without it a
    GPU loop runs ahead of the host and the iteration count overstates throughput. The
    window overshoots by at most one call; elapsed reports the measured value.
    """
    for _ in range(warmup):
        call(imgs)
        sync()
    times = []
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter()
        call(imgs)
        sync()
        times.append(time.perf_counter() - t)
        if time.perf_counter() - t0 >= duration:
            break
    return {"calls": len(times), "elapsed": time.perf_counter() - t0, "ms": np.array(times) * 1e3}


def describe_device(device: str) -> str:
    """Human-readable device line + torch version, so results are reproducible."""
    import torch

    if device.startswith("cuda"):
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        return (
            f"device    : cuda ({name}, compute {major}.{minor}) "
            f"| torch {torch.__version__} (cu{torch.version.cuda})"
        )
    if device == "mps":
        return f"device    : mps (Apple Silicon) | torch {torch.__version__}"
    return f"device    : cpu ({platform.processor() or platform.machine()}) | torch {torch.__version__}"


def main(
    model: str = "s",
    device: Optional[str] = None,
    duration: float = 5.0,
    warmup: int = 5,
    batch: int = 1,
    img_size: Optional[int] = None,
) -> int:
    """Benchmark `model` on the best available device; returns the process exit code."""
    from dfine_seg import load_model

    if model not in _SIZES and not model.endswith(".pt"):
        print(
            "hw_bench benchmarks the Torch path: pass a size (n|s|m|l|x) or a .pt file",
            file=sys.stderr,
        )
        return 1

    kwargs: Dict[str, Any] = {"conf_thresh": CONF_THRESH}
    if device:
        kwargs["device"] = device
    if img_size:
        kwargs["input_width"] = kwargs["input_height"] = img_size
    m = load_model(model, **kwargs)

    dev = getattr(m, "device", device or "cpu")
    img = make_input()
    imgs = img[None] if batch == 1 else np.repeat(img[None], batch, axis=0)

    def sync() -> None:
        pass

    if dev.startswith("cuda"):
        import torch

        sync = torch.cuda.synchronize
    elif dev == "mps":
        import torch

        sync = torch.mps.synchronize

    res = run_bench(lambda x: m(x, bgr=True), imgs, duration=duration, warmup=warmup, sync=sync)
    ms: np.ndarray = res["ms"]
    images = res["calls"] * batch
    fps = images / res["elapsed"]
    calls_s = res["calls"] / res["elapsed"]
    height, width = getattr(m, "input_size", (640, 640))  # (H, W)

    print(describe_device(dev))
    print(
        f"model     : {Path(model).name} ({getattr(m, 'model_name', '?')}) "
        f"| {getattr(m, 'task', 'detect')} | {height}x{width} fp32 | conf {CONF_THRESH}"
    )
    print(f"input     : synthetic {WIDTH}x{HEIGHT} noise (seed {SEED}) | batch {batch}")
    print(f"window    : {duration:g}s after {warmup} warmup calls")
    print()
    if batch == 1:
        print(
            f"{res['calls']} calls in {res['elapsed']:.2f}s  ->  {fps:.1f} images/s "
            f"({calls_s:.1f} calls/s)"
        )
    else:
        print(
            f"{res['calls']} calls x {batch} images = {images} in {res['elapsed']:.2f}s  "
            f"->  {fps:.1f} images/s ({calls_s:.1f} calls/s)"
        )
    print(
        f"per call  : mean {ms.mean():.1f} ms | "
        f"p50 {np.percentile(ms, 50):.1f} | p95 {np.percentile(ms, 95):.1f}"
    )
    return 0
