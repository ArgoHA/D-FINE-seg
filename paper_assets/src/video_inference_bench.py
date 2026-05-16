import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory
from pathlib import Path
from queue import Queue
from shutil import rmtree
from threading import Thread

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tabulate import tabulate
from tqdm import tqdm

from src.dl.utils import Visualizer, get_latest_experiment_name
from src.infer.trt_model import TRT_model


def open_capture(source) -> cv2.VideoCapture:
    s = str(source)
    cap = cv2.VideoCapture(int(s)) if s.isdigit() else cv2.VideoCapture(s)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source: {source}")
    return cap


def total_frames(cap: cv2.VideoCapture, max_frames: int | None) -> int | None:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:
        return min(n, max_frames) if n else max_frames
    return n or None


def run_base(model, visualizer, source, save_dir: Path, max_frames: int | None = None) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = open_capture(source)
    pbar = tqdm(total=total_frames(cap, max_frames))
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model(frame)[0]
            drawn = visualizer.draw(frame, res)
            cv2.imwrite(str(save_dir / f"{n:06d}.jpg"), drawn)
            n += 1
            pbar.update(1)
            if max_frames and n >= max_frames:
                break
    finally:
        cap.release()
        pbar.close()
    return n


def run_optimized(model, visualizer, source, save_dir: Path, max_frames: int | None = None) -> int:
    """
    Reader thread -> main inference -> drawer thread.
    Drawer also writes the rendered frame to disk.
    Pipeline rate = max(decode, infer, draw + imwrite).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = open_capture(source)

    read_q: Queue = Queue(maxsize=2)
    draw_q: Queue = Queue(maxsize=2)
    SENTINEL = None

    def reader():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_q.put((i, frame))
            i += 1
            if max_frames and i >= max_frames:
                break
        read_q.put(SENTINEL)

    def drawer_loop():
        while True:
            item = draw_q.get()
            if item is SENTINEL:
                break
            i, frame, res = item
            drawn = visualizer.draw(frame, res)
            cv2.imwrite(str(save_dir / f"{i:06d}.jpg"), drawn)

    t_r = Thread(target=reader, daemon=True)
    t_d = Thread(target=drawer_loop, daemon=True)
    t_r.start()
    t_d.start()

    pbar = tqdm(total=total_frames(cap, max_frames))
    n = 0
    try:
        while True:
            item = read_q.get()
            if item is SENTINEL:
                break
            i, frame = item
            res = model(frame)[0]
            draw_q.put((i, frame, res))
            n += 1
            pbar.update(1)
        draw_q.put(SENTINEL)
        t_r.join()
        t_d.join()
    finally:
        cap.release()
        pbar.close()
    return n


def run_optimized_v2(
    model,
    visualizer,
    source,
    save_dir: Path,
    max_frames: int | None = None,
    cpu_frac: float = 0.8,
    n_draw: int | None = None,
) -> int:
    """
    Pooled drawing:
      - 1 reader thread (VideoCapture is serial)
      - main thread runs inference and dispatches to drawers
      - n_draw worker threads draw and imwrite in parallel
    n_draw is derived from os.cpu_count(), which on Linux/macOS counts logical
    (SMT) threads — on a 6c/12t CPU this gives 8 drawers at cpu_frac=0.8.
    """
    if n_draw is None:
        n_draw = max(1, int((os.cpu_count() or 4) * cpu_frac) - 1)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = open_capture(source)
    read_q: Queue = Queue(maxsize=2 * n_draw + 2)
    draw_q: Queue = Queue(maxsize=2 * n_draw + 2)
    READ_DONE = object()
    DRAW_DONE = object()

    def reader():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_q.put((i, frame))
            i += 1
            if max_frames and i >= max_frames:
                break
        read_q.put(READ_DONE)

    def drawer():
        while True:
            item = draw_q.get()
            if item is DRAW_DONE:
                return
            i, frame, res = item
            drawn = visualizer.draw(frame, res)
            cv2.imwrite(str(save_dir / f"{i:06d}.jpg"), drawn)

    t_r = Thread(target=reader, daemon=True)
    drawers = [Thread(target=drawer, daemon=True) for _ in range(n_draw)]
    t_r.start()
    for t in drawers:
        t.start()

    pbar = tqdm(total=total_frames(cap, max_frames))
    n = 0
    try:
        while True:
            item = read_q.get()
            if item is READ_DONE:
                break
            i, frame = item
            res = model(frame)[0]
            draw_q.put((i, frame, res))
            n += 1
            pbar.update(1)
        for _ in drawers:
            draw_q.put(DRAW_DONE)
        for t in drawers:
            t.join()
        t_r.join()
    finally:
        cap.release()
        pbar.close()
    return n


def _drawer_proc(
    in_q: "mp.Queue",
    free_q: "mp.Queue",
    save_dir_str: str,
    n_classes: int,
    class_names: dict,
    frame_shape: tuple,
    frame_dtype_str: str,
) -> None:
    """
    Worker entrypoint for run_optimized_v3. Reads frames from a pool of
    shared-memory blocks (attached by name) instead of from pickled queue
    payloads. Returns the block to free_q as soon as visualizer.draw is
    done with it (draw copies internally).
    """
    visualizer = Visualizer(n_classes=n_classes, class_names=class_names)
    save_dir = Path(save_dir_str)
    dtype = np.dtype(frame_dtype_str)
    shm_cache: dict[str, shared_memory.SharedMemory] = {}
    view_cache: dict[str, np.ndarray] = {}
    try:
        while True:
            item = in_q.get()
            if item is None:
                return
            i, shm_name, res = item
            view = view_cache.get(shm_name)
            if view is None:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm_cache[shm_name] = shm
                view = np.ndarray(frame_shape, dtype=dtype, buffer=shm.buf)
                view_cache[shm_name] = view
            drawn = visualizer.draw(view, res)
            free_q.put(shm_name)
            cv2.imwrite(str(save_dir / f"{i:06d}.jpg"), drawn)
    finally:
        for shm in shm_cache.values():
            shm.close()


def run_optimized_v3(
    model,
    visualizer,
    source,
    save_dir: Path,
    max_frames: int | None = None,
    cpu_frac: float = 0.8,
    n_draw: int | None = None,
) -> int:
    """
    Multiprocessing variant of run_optimized_v2, with shared memory for frames.
      - 1 reader thread (in main process)
      - main process runs inference, memcpys the frame into a free shm block,
        and ships only (idx, shm_name, res) over the mp.Queue
      - n_draw worker processes attach to the shm block by name, draw, imwrite,
        and return the block to the free pool

    Removes the pickling tax of a naive multiprocessing version (which would
    serialize every ~6MB frame through a pipe). Only the small result dict
    crosses the queue; the frame travels through a fixed pool of shm blocks.
    """
    if n_draw is None:
        n_draw = max(1, int((os.cpu_count() or 4) * cpu_frac) - 1)
    save_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    cap = open_capture(source)

    # Peek the first frame to learn shape + dtype + nbytes for the shm pool.
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        return 0
    frame_shape = first_frame.shape
    frame_dtype_str = first_frame.dtype.str
    nbytes = first_frame.nbytes

    pool_size = 2 * n_draw + 4
    shms = [shared_memory.SharedMemory(create=True, size=nbytes) for _ in range(pool_size)]
    free_q: "mp.Queue" = ctx.Queue()
    for shm in shms:
        free_q.put(shm.name)

    draw_q: "mp.Queue" = ctx.Queue(maxsize=2 * n_draw + 2)
    read_q: Queue = Queue(maxsize=2 * n_draw + 2)
    READ_DONE = object()

    def reader():
        read_q.put((0, first_frame))
        i = 1
        while True:
            if max_frames and i >= max_frames:
                break
            ok2, frame = cap.read()
            if not ok2:
                break
            read_q.put((i, frame))
            i += 1
        read_q.put(READ_DONE)

    n_classes = len(visualizer.class_names)
    class_names = dict(visualizer.class_names)
    save_dir_str = str(save_dir)

    drawers = [
        ctx.Process(
            target=_drawer_proc,
            args=(
                draw_q,
                free_q,
                save_dir_str,
                n_classes,
                class_names,
                frame_shape,
                frame_dtype_str,
            ),
            daemon=True,
        )
        for _ in range(n_draw)
    ]
    for p in drawers:
        p.start()

    t_r = Thread(target=reader, daemon=True)
    t_r.start()

    # Producer-side: keep a numpy view onto every owned shm block so writes
    # are a single memcpy with no re-attach cost.
    views = {
        s.name: np.ndarray(frame_shape, dtype=np.dtype(frame_dtype_str), buffer=s.buf) for s in shms
    }

    pbar = tqdm(total=total_frames(cap, max_frames))
    n = 0
    try:
        while True:
            item = read_q.get()
            if item is READ_DONE:
                break
            i, frame = item
            res = model(frame)[0]
            shm_name = free_q.get()  # blocks if all blocks are in flight (backpressure)
            views[shm_name][...] = frame
            draw_q.put((i, shm_name, res))
            n += 1
            pbar.update(1)
        for _ in drawers:
            draw_q.put(None)
        for p in drawers:
            p.join()
        t_r.join()
    finally:
        cap.release()
        pbar.close()
        for shm in shms:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
    return n


def profile_stages(model, visualizer, source, save_dir: Path, n_frames: int = 200) -> None:
    """
    Run each stage in isolation and print ms/frame.
    Now includes imwrite so we can see disk-write cost too.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = open_capture(source)
    frames: list = []
    t0 = time.perf_counter()
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    decode_ms = 1000 * (time.perf_counter() - t0) / max(len(frames), 1)
    cap.release()
    if not frames:
        print("profile_stages: could not read any frames")
        return

    t0 = time.perf_counter()
    results = [model(f)[0] for f in frames]
    infer_ms = 1000 * (time.perf_counter() - t0) / len(frames)

    t0 = time.perf_counter()
    drawn = [visualizer.draw(f, r) for f, r in zip(frames, results)]
    draw_ms = 1000 * (time.perf_counter() - t0) / len(frames)

    t0 = time.perf_counter()
    for i, d in enumerate(drawn):
        cv2.imwrite(str(save_dir / f"{i:06d}.jpg"), d)
    write_ms = 1000 * (time.perf_counter() - t0) / len(drawn)

    rows = [
        ("decode", round(decode_ms, 2), round(1000 / decode_ms, 1) if decode_ms else 0),
        ("infer", round(infer_ms, 2), round(1000 / infer_ms, 1) if infer_ms else 0),
        ("draw", round(draw_ms, 2), round(1000 / draw_ms, 1) if draw_ms else 0),
        ("imwrite", round(write_ms, 2), round(1000 / write_ms, 1) if write_ms else 0),
    ]
    print(
        f"\nper-stage profile over {len(frames)} frames:\n"
        + tabulate(rows, headers=["stage", "ms/frame", "max fps"], tablefmt="pretty")
    )


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    source = Path(cfg.train.path_to_test_data) / "test.mp4"
    max_frames = cfg.train.get("max_frames", None)

    model = TRT_model(
        model_path=Path(cfg.train.path_to_save) / "model.engine",
        n_outputs=len(cfg.train.label_to_name),
        conf_thresh=0.5,
        rect=False,
        keep_ratio=cfg.train.keep_ratio,
        apply_nms=True,
    )

    visualizer = Visualizer(
        n_classes=len(cfg.train.label_to_name), class_names=cfg.train.label_to_name
    )

    output_dir = Path(cfg.train.infer_path)
    profile_dir = output_dir / "profile"
    if profile_dir.exists():
        rmtree(profile_dir)
    profile_stages(model, visualizer, source, profile_dir)

    results: list[tuple[str, int, float, float]] = []

    def time_run(name: str, fn, stem: str, **kwargs):
        save_dir = output_dir / stem
        if save_dir.exists():
            rmtree(save_dir)
        t0 = time.perf_counter()
        n = fn(
            model=model,
            visualizer=visualizer,
            source=source,
            save_dir=save_dir,
            max_frames=max_frames,
            **kwargs,
        )
        dt = time.perf_counter() - t0
        results.append((name, n, round(dt, 2), round(n / dt, 1) if dt > 0 else 0.0))

    time_run("Sequential", run_base, "sequential")
    time_run("Pipelined (1r/1d)", run_optimized, "pipelined")

    n_workers = max(2, int((os.cpu_count() or 4) * 0.8))
    n_draw = max(1, n_workers - 1)
    time_run(f"Pooled threads (1r/{n_draw}d)", run_optimized_v2, "pooled", n_draw=n_draw)
    # time_run(f"Pooled procs (1r/{n_draw}d)", run_optimized_v3, "pooled_mp", n_draw=n_draw)

    print(
        "\n"
        + tabulate(
            results,
            headers=["run", "frames", "time (s)", "fps"],
            tablefmt="pretty",
        )
    )


if __name__ == "__main__":
    main()
