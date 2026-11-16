"""
D-FINE-seg demo - a small browser UI for detection, instance and semantic segmentation,
keypoints and SAM3. Standard library only: `http.server` serves the page in
`dfine_seg/app/demo_ui/` and a JSON / raw-bytes API.

    dfine demo          (or: python -m dfine_seg.app.demo)

COCO detection `s` loads at startup; everything else is picked in the page:
  D-FINE-seg - a released preset (n|s|m|l|x per task) or your own artifact, by extension:
    .pt -> PyTorch (CUDA / MPS / CPU), .engine -> TensorRT, .onnx -> ONNX Runtime,
    .xml -> OpenVINO
  SAM3       - text-promptable instance segmentation (facebook/sam3, lazy-loaded;
               needs `pip install 'dfine-seg[label]'`)
Inputs: an image (upload, drop or paste), the webcam (live, frame by frame) or a video file
(rendered to H.264 with ffmpeg when it is installed).
"""

import importlib.util
import ipaddress
import json
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from dfine_seg import load_model
from dfine_seg.api.loader import SIZES
from dfine_seg.viz import Visualizer, classes_from_model

DEFAULT_MODEL = "s"  # size (n|s|m|l|x) -> released weights, or a path to an artifact
DEFAULT_TASK = "auto"  # auto | detect | segment | sem_seg | kpts
DEFAULT_CONF_THRESH = 0.5
UI_DIR = Path(__file__).with_name("demo_ui")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}

# Released checkpoints, label -> load_model(size, task). Anything else is treated as a path.
TASKS = ("detect", "segment", "sem_seg", "kpts")
PRESETS = {f"{s} ({t})": (s, t) for t in TASKS for s in SIZES}
WEIGHT_SUFFIXES = (".pt", ".onnx", ".engine")  # OpenVINO needs its .bin sibling: use a path

SAM3_MODEL_ID = "facebook/sam3"
SAM3_AVAILABLE = importlib.util.find_spec("transformers") is not None


# ─── State ───────────────────────────────────────────────────────────────
@dataclass
class Loaded:
    """The D-FINE backend currently serving every input."""

    model: object = None
    # Fixed at load time, not per frame: colors come from the model's class count, not
    # from the labels present in one frame, which would repaint classes as the scene changes.
    vis: Optional[Visualizer] = None
    info: dict = field(default_factory=dict)


CURRENT = Loaded()
# One forward at a time: webcam frames, image runs and video jobs share one GPU model.
INFER_LOCK = threading.Lock()
LOAD_LOCK = threading.Lock()  # one load at a time; INFER_LOCK is held only for the swap
SAM_LOCK = threading.Lock()
_sam_model = None
WORK_DIR: Optional[Path] = None  # uploads and video results; created by main()


def parse_names(text: str) -> Optional[Dict[int, str]]:
    """`person, car` or one per line -> {0: person, 1: car}. `3: dog` pins an explicit id."""
    items = [t.strip() for line in (text or "").splitlines() for t in line.split(",")]
    names, nxt = {}, 0
    for item in filter(None, items):
        idx, sep, name = item.partition(":")
        if sep and idx.strip().isdigit():
            idx, item = int(idx), name.strip()
        else:
            idx = nxt
        names[idx], nxt = item, idx + 1
    return names or None


def parse_size(text: str) -> Tuple[int, int]:
    """`640` -> (640, 640); `1024x2048` / `1024, 2048` -> (1024, 2048)."""
    parts = re.split(r"[x,\s]+", str(text).strip())  # unfiltered: `12x` must not read as 12
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"input size must be `640` or `1024x2048`, got {text!r}")
    return int(parts[0]), int(parts[1])


def load_backend(spec: str, names_text: str = "", input_size: str = "", task: str = "auto"):
    """(Re)load the D-FINE backend. Raises on failure and keeps the working model."""
    spec = (spec or "").strip() or DEFAULT_MODEL
    if spec in SIZES:  # `dfine demo m --task segment` -> its preset entry
        spec = f"{spec} ({'detect' if task == 'auto' else task})"
    src, preset_task = PRESETS.get(spec, (spec, None))  # a preset carries its own task
    task = preset_task or task
    suffix = Path(src).suffix.lower()

    kwargs = {"conf_thresh": DEFAULT_CONF_THRESH}
    # Blank = let the wrapper decide: a .pt reads train.img_size off the config frozen beside
    # it, graph artifacts read the graph.
    if suffix in ("", ".pt") and str(input_size).strip():
        kwargs["input_height"], kwargs["input_width"] = parse_size(input_size)
    # task picks the weights for a size preset and the architecture for a .pt; graph
    # artifacts have it baked in, and their wrappers take no task=.
    picked = None if task == "auto" or suffix not in ("", ".pt") else task

    # Load outside INFER_LOCK: a first-use download or engine build must not stall
    # running webcam frames and video jobs.
    with LOAD_LOCK:
        model = load_model(src, task=picked, names=parse_names(names_text), **kwargs)
        known, names = classes_from_model(model)
        h, w = getattr(model, "input_size", (None, None))
        vis = Visualizer(model)
        info = {
            "spec": spec,
            "preset": spec in PRESETS,
            "name": spec if spec in PRESETS else Path(src).name,
            "backend": type(model).__name__,
            "task": getattr(model, "task", None),
            "classes": known,
            "named": len(names),
            "device": str(getattr(model, "device", "")) or None,
            "input": f"{h}x{w}" if h else None,
        }
        with INFER_LOCK:
            CURRENT.model, CURRENT.vis, CURRENT.info = model, vis, info
    print(f"[load] {json.dumps(CURRENT.info)}", flush=True)
    return CURRENT.info


def _get_sam_model():
    """Lazy-load SAM3 on first use - seconds from the HF cache, a ~6.5 GB download without."""
    global _sam_model
    with SAM_LOCK:
        if _sam_model is None:
            from dfine_seg.infer.sam3_model import SAM3Model

            print(f"Loading {SAM3_MODEL_ID} (~6.5 GB download if not cached) …", flush=True)
            t0 = time.perf_counter()
            _sam_model = SAM3Model(model_path=SAM3_MODEL_ID, conf_thresh=DEFAULT_CONF_THRESH)
            print(f"Loaded {SAM3_MODEL_ID} in {time.perf_counter() - t0:.1f}s", flush=True)
    return _sam_model


@dataclass
class Run:
    """Settings of one image / webcam frame / video job, parsed from the query string."""

    backend: str = "dfine"  # dfine | sam3
    prompt: str = "person"
    conf: float = DEFAULT_CONF_THRESH
    minimize: bool = False
    stride: int = 1

    @classmethod
    def from_query(cls, q: dict) -> "Run":
        one = {k: v[-1] for k, v in q.items()}
        return cls(
            backend="sam3" if one.get("backend") == "sam3" else "dfine",
            prompt=one.get("prompt", "person"),
            conf=float(np.clip(float(one.get("conf", DEFAULT_CONF_THRESH)), 0.0, 1.0)),
            minimize=one.get("minimize") in ("1", "true"),
            stride=max(1, int(one.get("stride", 1))),
        )

    def predictor(self):
        """-> (callable frame -> result dict, Visualizer), both fixed for this run."""
        if self.backend == "sam3":
            if not SAM3_AVAILABLE:
                raise RuntimeError("SAM3 needs transformers: pip install 'dfine-seg[label]'")
            model = _get_sam_model()
            prompts = model.parse_prompts(self.prompt)
            vis = Visualizer(n_classes=len(prompts), class_names=dict(enumerate(prompts)))

            def predict(img):
                with INFER_LOCK:
                    # Prompts persist on the shared model; setting them drops the cached text
                    # embeddings, so only do it when another run changed them.
                    if model.prompts != prompts:
                        model.prompts = prompts
                    return model(img, conf_thresh=self.conf)[0]

            return predict, vis

        model, vis = CURRENT.model, CURRENT.vis
        if model is None:
            raise RuntimeError("no model loaded")

        def predict(img):
            with INFER_LOCK:
                return model(img, conf_thresh=self.conf)[0]

        return predict, vis


def _summary(out: dict) -> str:
    if "sem_seg" in out:
        labels = out["sem_seg"]
        labels = np.asarray(labels.cpu() if hasattr(labels, "cpu") else labels)
        n = len(np.unique(labels[labels != 255]))
        return f"{n} class{'es' * (n != 1)}"
    n = len(out.get("labels", []))
    return f"{n} object{'s' * (n != 1)}"


def annotate(img: np.ndarray, run: Run) -> Tuple[np.ndarray, float, str]:
    """BGR frame -> (annotated BGR, model ms, summary)."""
    predict, vis = run.predictor()
    t0 = time.perf_counter()
    out = predict(img)
    ms = (time.perf_counter() - t0) * 1000
    return vis(img, out, minimize=run.minimize), ms, _summary(out)


# ─── Video jobs ──────────────────────────────────────────────────────────
@dataclass
class Job:
    src: Path
    run: Run
    out: Path = None
    status: str = "queued"  # queued | running | encoding | done | error | cancelled
    frames: int = 0
    total: int = 0
    fps: float = 0.0
    error: str = ""
    preview: bytes = b""  # latest annotated frame (JPEG) for the live preview
    playable: bool = True  # False = mp4v fallback without ffmpeg, download only
    cancel: bool = False

    def public(self) -> dict:
        keys = ("status", "frames", "total", "fps", "error", "playable")
        return {k: getattr(self, k) for k in keys}


JOBS: Dict[str, Job] = {}


class VideoWriter:
    """Pipes raw frames into ffmpeg (H.264 in one pass, so browsers can play the result);
    without ffmpeg it falls back to OpenCV's mp4v, which browsers cannot play."""

    def __init__(self, path: Path, fps: float, w: int, h: int):
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
                "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even sides
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ]  # fmt: skip
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> bool:
        """Finish the file; returns whether a browser can play it."""
        if not self.proc:
            self.cv.release()
            return False
        _, err = self.proc.communicate()
        if self.proc.returncode:
            raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')[-300:]}")
        return True

    def abort(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        elif self.cv:
            self.cv.release()


def run_video(job: Job) -> None:
    """Process every `stride`-th frame; skipped frames reuse the last result."""
    cap, writer = cv2.VideoCapture(str(job.src)), None
    try:
        if not cap.isOpened():
            raise RuntimeError("cannot open this video")
        predict, vis = job.run.predictor()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job.total, job.status = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "running"
        writer = VideoWriter(job.out, fps, w, h)
        last, t0, shown = None, time.perf_counter(), 0.0
        while not job.cancel:
            ok, frame = cap.read()
            if not ok:
                break
            if job.frames % job.run.stride == 0:
                last = predict(frame)
            frame = vis(frame, last, minimize=job.run.minimize)
            writer.write(frame)
            job.frames += 1
            now = time.perf_counter()
            job.fps = job.frames / (now - t0)
            if now - shown > 0.25:  # a preview a few times per second is enough
                job.preview, shown = _jpeg(frame, 80), now
        job.status = "encoding"
        job.playable, writer = writer.close(), None
        job.status = "cancelled" if job.cancel else "done"
        print(f"[video] {job.status}: {job.frames} frames, {job.fps:.1f} fps", flush=True)
    except Exception as e:
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
        print(f"[video] {job.error}", flush=True)
        if writer:
            writer.abort()
    finally:
        cap.release()


def start_job(src: Path, run: Run) -> str:
    """Start a video job. Finished jobs' results are deleted to keep the temp dir small;
    their sources are kept while the new job uses them, so "process again" needs no upload."""
    for jid, old in list(JOBS.items()):
        if old.status not in ("queued", "running", "encoding"):
            JOBS.pop(jid).out.unlink(missing_ok=True)
            if old.src != src:
                old.src.unlink(missing_ok=True)
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = job = Job(src=src, run=run, out=WORK_DIR / f"{jid}.mp4")
    threading.Thread(target=run_video, args=(job,), daemon=True).start()
    return jid


def _jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


# ─── HTTP ────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "dfine-demo"

    def log_message(self, fmt, *args):  # quiet: webcam mode is a request per frame
        pass

    # responses
    def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK, headers: dict = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def _error(self, msg: str, status=HTTPStatus.BAD_REQUEST):
        self._json({"error": msg}, status)

    def _file(self, path: Path, ctype: str):
        """Serve a file with single-range support, so the browser can seek in the video."""
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            if m[1]:
                start, end = int(m[1]), min(int(m[2] or end), end)
            else:  # suffix range: the last N bytes
                start = max(size - int(m[2]), 0)
            if start > end:  # past EOF, inverted, or `bytes=-0`
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        partial = end - start + 1 < size
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0 and (chunk := f.read(min(1 << 20, left))):
                self.wfile.write(chunk)
                left -= len(chunk)

    # requests
    def _host_ok(self) -> bool:
        """DNS-rebinding guard: a rebound attacker domain makes its page same-origin (so the
        X-DFine header no longer helps), but its requests still carry the attacker's Host.
        Accept IP literals, localhost and this machine's own names only."""
        host = (urlsplit(f"//{self.headers.get('Host', '')}").hostname or "").lower()
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return host in self.server.allowed_hosts

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _body_to(self, path: Path) -> Path:
        left = int(self.headers.get("Content-Length", 0))
        with open(path, "wb") as f:
            while left > 0 and (chunk := self.rfile.read(min(1 << 20, left))):
                f.write(chunk)
                left -= len(chunk)
        return path

    def do_GET(self):
        if not self._host_ok():
            return self._error("unexpected Host header", HTTPStatus.FORBIDDEN)
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        if url.path in STATIC:
            name, ctype = STATIC[url.path]
            return self._send((UI_DIR / name).read_bytes(), ctype)
        if url.path == "/api/state":
            return self._json(
                {
                    "presets": [{"key": k, "size": s, "task": t} for k, (s, t) in PRESETS.items()],
                    "model": CURRENT.info,
                    "sam3": SAM3_AVAILABLE,
                    "ffmpeg": shutil.which("ffmpeg") is not None,
                }
            )
        if len(parts) >= 3 and parts[:2] == ["api", "video"] and parts[2] in JOBS:
            job = JOBS[parts[2]]
            if len(parts) == 3:
                return self._json(job.public())
            if parts[3] == "preview" and job.preview:
                return self._send(job.preview, "image/jpeg")
            if parts[3] == "result" and job.status in ("done", "cancelled"):
                return self._file(job.out, "video/mp4")
        self._error("not found", HTTPStatus.NOT_FOUND)

    do_HEAD = do_GET

    def do_POST(self):
        if not self._host_ok():
            return self._error("unexpected Host header", HTTPStatus.FORBIDDEN)
        # A custom header forces a CORS preflight that this server never answers, so other
        # websites open in the same browser cannot drive the API (it loads local files).
        if self.headers.get("X-DFine") != "1":
            return self._error("missing X-DFine header", HTTPStatus.FORBIDDEN)
        url = urlsplit(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/api/load":
                req = json.loads(self._body() or b"{}")
                info = load_backend(
                    req.get("model", ""), req.get("names", ""), req.get("input_size", "")
                )
                return self._json({"model": info})
            if url.path == "/api/upload":  # weights, raw body; the client then calls /api/load
                suffix = Path(q.get("name", [""])[-1]).suffix.lower()
                if suffix not in WEIGHT_SUFFIXES:
                    return self._error(f"expected one of {', '.join(WEIGHT_SUFFIXES)}")
                stem = re.sub(r"[^\w.-]", "_", Path(q["name"][-1]).stem)[:60] or "weights"
                (WORK_DIR / "weights").mkdir(exist_ok=True)
                path = self._body_to(WORK_DIR / "weights" / f"{stem}{suffix}")
                return self._json({"path": str(path)})
            if url.path == "/api/image":
                buf = np.frombuffer(self._body(), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
                if img is None:
                    return self._error("could not decode the image")
                run = Run.from_query(q)
                vis, ms, summary = annotate(img, run)
                quality = int(q.get("quality", ["92"])[-1])
                headers = {"X-Infer-Ms": f"{ms:.1f}", "X-Summary": summary}
                return self._send(_jpeg(vis, quality), "image/jpeg", headers=headers)
            if url.path == "/api/video":
                src = self._body_to(WORK_DIR / f"{uuid.uuid4().hex[:12]}_in")
                return self._json({"job": start_job(src, Run.from_query(q))})
            parts = url.path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "video"] and parts[2] in JOBS:
                job = JOBS[parts[2]]
                if parts[3] == "cancel":
                    job.cancel = True
                    return self._json({"ok": True})
                if parts[3] == "rerun":  # same uploaded source, the settings in the query
                    if not job.src.exists():
                        return self._error("the source video is gone, open it again")
                    return self._json({"job": start_job(job.src, Run.from_query(q))})
        except Exception as e:  # report to the page; the server and current model stay up
            return self._error(f"{type(e).__name__}: {e}", HTTPStatus.UNPROCESSABLE_ENTITY)
        self._error("not found", HTTPStatus.NOT_FOUND)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    name = socket.gethostname().lower()
    server.allowed_hosts = {"localhost", host.lower(), name, f"{name}.local"}
    return server


def main(
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    host: str = "0.0.0.0",  # LAN-reachable: the page loads arbitrary local paths as models
    port: int = 7860,
) -> None:
    global WORK_DIR
    # Load before serving, so the page opens ready to run.
    load_backend(model, task=task)
    server = make_server(host, port)
    WORK_DIR = Path(tempfile.mkdtemp(prefix="dfine_demo_"))  # after anything that can fail
    if host not in ("127.0.0.1", "localhost"):
        # The model panel loads any path the browser sends, so this hands local-file
        # probing and untrusted-graph deserialization to everyone who can reach the port.
        print(
            f"WARNING: serving on {host} - anyone who can reach this port can load "
            "any file on this machine as a model"
        )
    # 0.0.0.0 is a bind address, not a thing a browser can open.
    click = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Open http://{click}:{server.server_port} in your browser", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
