"""
D-FINE-seg Gradio Demo - detection, instance segmentation, semantic segmentation

Just run it - COCO detection weights download on first use:
    dfine demo          (or: python -m dfine_seg.app.demo)

Everything is set from the UI; nothing here needs editing. The "Model" panel swaps in
your own checkpoint at runtime (size preset or a path/upload) and lets you name its
classes, so a freshly trained model can be tried on your images and videos immediately.

Backends selectable in the UI:
  D-FINE-seg - size preset (n|s|m|l|x) or a local artifact, format picked by extension:
    .pt      -> PyTorch   (CUDA / MPS / CPU)
    .engine  -> TensorRT  (CUDA)
    .onnx    -> ONNXRuntime
    .xml     -> OpenVINO  (CPU / iGPU)
  SAM3       - text-promptable instance segmentation (facebook/sam3, lazy-loaded)

Tabs:
  1. Images - upload or webcam snapshot -> annotated result
  2. Video  - upload a video file -> annotated output
"""

import re
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from dfine_seg import load_model
from dfine_seg.api.loader import SIZES
from dfine_seg.viz import Visualizer, overlay_sem_seg, sem_seg_palette

# ─── Startup defaults (all overridable in the UI) ───────────────────────
DEFAULT_MODEL = "s"  # size (n|s|m|l|x) -> COCO weights, or a path to .pt/.engine/.onnx/.xml
DEFAULT_TASK = "auto"  # auto | detect | segment | sem_seg
DEFAULT_INPUT_SIZE = ""  # blank: a .pt reads its own config, a graph reads the graph
DEFAULT_CONF_THRESH = 0.5  # initial slider value
# ─────────────────────────────────────────────────────────────────────────

# The 10 released COCO checkpoints, label -> load_model(size, task). Labels mirror the
# Python call, and anything not in here is treated as a path, so one field covers both.
PRESETS = {f"{s} ({t})": (s, t) for t in ("detect", "segment") for s in SIZES}


# ─── Model loading (driven by the UI) ────────────────────────────────────
@dataclass
class Loaded:
    """The D-FINE backend currently serving both tabs."""

    model: object = None
    vis: Optional[Visualizer] = None
    names: Dict[int, str] = field(default_factory=dict)
    # Fixed at load time, not per frame: deriving it from the labels present in one frame
    # repaints every class as the scene changes.
    palette: Optional[np.ndarray] = None


CURRENT = Loaded()


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
    """`640` -> (640, 640); `1024x2048` / `1024, 2048` -> (1024, 2048). Blank is handled above."""
    parts = re.split(r"[x,\s]+", str(text).strip())  # unfiltered: `12x` must not read as 12
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"input size must be `640` or `1024x2048`, got {text!r}")
    return int(parts[0]), int(parts[1])


def load_backend(spec: str, names_text: str, input_size: str, task: str = "auto") -> str:
    """(Re)load the D-FINE backend from the UI controls; returns a status line."""
    src = (spec or "").strip() or DEFAULT_MODEL
    src, preset_task = PRESETS.get(src, (src, None))  # a preset carries its own task
    task = preset_task or task
    suffix = Path(src).suffix.lower()
    given = parse_names(names_text)

    kwargs = {"conf_thresh": DEFAULT_CONF_THRESH}
    # Blank means "let the wrapper decide": a .pt reads train.img_size off the config frozen
    # beside it, graph artifacts read the graph. Only override when asked.
    if suffix in ("", ".pt") and str(input_size).strip():  # "" = a size preset
        try:
            kwargs["input_height"], kwargs["input_width"] = parse_size(input_size)
        except ValueError as e:
            return f"❌ {e}"
    # task selects the weights for a size preset and the architecture for a .pt; graph
    # artifacts have it baked in, and their wrappers take no task=.
    picked = None if task == "auto" or suffix not in ("", ".pt") else task

    try:
        model = load_model(src, task=picked, names=given, **kwargs)
    except Exception as e:  # keep the working model rather than leaving the demo dead
        kept = f" - keeping {Path(CURRENT.model.model_path).name}" if CURRENT.model else ""
        return f"❌ {type(e).__name__}: {e}{kept}"

    names = model.names or {}
    # Fused-postprocess graphs (.onnx/.engine/.mlpackage) carry no class count anywhere, so
    # their wrappers have no n_outputs at all and the names box is all we have.
    known = getattr(model, "n_outputs", 0) or 0
    known = max(known, max(names) + 1 if names else 0)
    CURRENT.model = model
    CURRENT.names = names
    CURRENT.vis = Visualizer(n_classes=known or 80, class_names=names or None)
    CURRENT.palette = sem_seg_palette(known or 80)

    h, w = getattr(model, "input_size", (None, None))
    note = f" ({len(names)} named)" if 0 < len(names) < known else ("" if names else " (unnamed)")
    # Plain text: this line is both the UI status and the console log, so no markup.
    return (
        f"✅ {type(model).__name__} | {Path(src).name} | "
        f"task: {getattr(model, 'task', 'from graph')} | "
        f"classes: {known or '? - name them above'}{note if known else ''} | "
        f"device: {getattr(model, 'device', '?')} | input: {h}x{w}"
    )


# ─── SAM3 (text-promptable) backend ─────────────────────────────────────
SAM3_MODEL_ID = "facebook/sam3"

_sam_model = None


def _get_sam_model():
    """Lazy-load SAM3 on first use - seconds from the HF cache, a ~6.5 GB download without."""
    global _sam_model
    if _sam_model is None:
        from dfine_seg.infer.sam3_model import SAM3Model

        gr.Info(f"Loading {SAM3_MODEL_ID} - downloads ~6.5 GB if it isn't cached yet")
        print(f"Loading {SAM3_MODEL_ID} …", flush=True)
        t0 = time.perf_counter()
        _sam_model = SAM3Model(model_path=SAM3_MODEL_ID, conf_thresh=DEFAULT_CONF_THRESH)
        print(f"Loaded {SAM3_MODEL_ID} in {time.perf_counter() - t0:.1f}s", flush=True)
    return _sam_model


# ─── Initialization ─────────────────────────────────────────────────────
DEFAULT_BACKEND = "D-FINE-seg"
sam_visualizer = Visualizer(n_classes=1)  # single prompt class; name set per-run


# ─── Inference helpers ───────────────────────────────────────────────────
def _set_model_conf_threshold(model, conf_thresh: float) -> None:
    """Set a uniform confidence threshold for the currently loaded backend."""
    conf = float(np.clip(conf_thresh, 0.0, 1.0))
    if getattr(model, "conf_threshs", None) is not None:
        model.conf_threshs = [conf] * len(model.conf_threshs)
        if getattr(model, "_conf_threshs_t", None) is not None:
            model._conf_threshs_t.fill_(conf)  # TRT reads this device copy, not the list
    elif hasattr(model, "conf_thresh"):
        model.conf_thresh = conf


def _select_backend(backend: str, prompt: str, conf_thresh: float):
    """Return (model, visualizer) for the chosen backend, applying conf / prompt."""
    if backend == "SAM3":
        m = _get_sam_model()
        m.prompt = (prompt or "object").strip()
        m.conf_thresh = float(np.clip(conf_thresh, 0.0, 1.0))
        sam_visualizer.class_names = {0: m.prompt}
        return m, sam_visualizer
    if CURRENT.model is None:
        raise gr.Error("No model loaded - fix the model settings above and press Load.")
    _set_model_conf_threshold(CURRENT.model, conf_thresh)
    return CURRENT.model, CURRENT.vis


def _run_on_bgr(img_bgr, model_obj, vis_obj, minimize: bool = False) -> np.ndarray:
    """Run model + visualizer on a single BGR frame. Returns annotated BGR."""
    return _draw(img_bgr, model_obj(img_bgr)[0], vis_obj, minimize=minimize)


def _draw(img_bgr, results: dict, vis_obj, minimize: bool = False) -> np.ndarray:
    """Boxes/masks, or a palette overlay when the model is dense (sem_seg)."""
    if "sem_seg" in results:
        return overlay_sem_seg(img_bgr, results["sem_seg"].cpu().numpy(), CURRENT.palette)
    return vis_obj.draw(img_bgr, results, minimize=minimize)


# ─── Tab 1: Images (single upload or webcam snapshot) ───────────────────
def predict_image(
    img: np.ndarray | None,
    backend: str = DEFAULT_BACKEND,
    prompt: str = "person",
    conf_thresh: float = DEFAULT_CONF_THRESH,
    minimize: bool = False,
):
    """Accept a single RGB image, return annotated RGB."""
    if img is None:
        return None
    # Logged before the work starts, so a slow click is distinguishable from a queued one.
    print(f"[image] {backend} {img.shape[1]}x{img.shape[0]} …", flush=True)
    model_obj, vis_obj = _select_backend(backend, prompt, conf_thresh)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    vis = _run_on_bgr(img_bgr, model_obj, vis_obj, minimize=minimize)
    ms = (time.perf_counter() - t0) * 1000
    print(f"[image] {backend} {ms:.1f} ms")
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ─── Tab 2: Video ───────────────────────────────────────────────────────
def predict_video(
    video_path: str | None,
    backend: str = DEFAULT_BACKEND,
    prompt: str = "person",
    conf_thresh: float = DEFAULT_CONF_THRESH,
    stride: int = 1,
    minimize: bool = False,
):
    """Process every `stride`-th frame; copy annotations to skipped frames."""
    if video_path is None:
        return None
    model_obj, vis_obj = _select_backend(backend, prompt, conf_thresh)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(stride))
    print(f"[video] {backend} {w}x{h}, {total} frames, stride {stride} …", flush=True)

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    idx = 0
    last_results = None
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            results = model_obj(frame)
            last_results = results[0]
        if last_results is not None:
            frame = _draw(frame, last_results, vis_obj, minimize=minimize)
        writer.write(frame)
        idx += 1
        if idx % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[video] {idx}/{total} frames  ({idx / elapsed:.1f} fps)")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"[video] done - {idx} frames in {elapsed:.1f}s ({idx / elapsed:.1f} fps)")

    # Re-encode to H.264 so browsers can play it
    h264_path = tempfile.mktemp(suffix=".mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                out_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                h264_path,
            ],
            check=True,
            capture_output=True,
        )
        Path(out_path).unlink(missing_ok=True)
        return h264_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[video] ffmpeg re-encode failed ({e}), returning mp4v file")
        return out_path


# ─── Build Gradio app ───────────────────────────────────────────────────
def build_ui(model: str = DEFAULT_MODEL, task: str = DEFAULT_TASK) -> gr.Blocks:
    """Build the app; loads the startup model first so the UI opens ready to run."""
    startup_status = load_backend(model, "", DEFAULT_INPUT_SIZE, task)
    print(startup_status)
    # Show the startup model as its preset entry when it is one, else as the raw path.
    wanted = (model, "detect" if task == "auto" else task)
    initial = next((label for label, v in PRESETS.items() if v == wanted), model)

    with gr.Blocks(title="D-FINE-seg + SAM3 Demo") as demo:
        gr.Markdown(
            f"# D-FINE-seg + SAM3 Demo\nSecond backend: `{SAM3_MODEL_ID}` (text-promptable)"
        )
        model_status = gr.Markdown(startup_status)  # outside the accordion: always visible

        with gr.Accordion("Change model", open=False) as model_panel:
            with gr.Row():
                model_spec = gr.Dropdown(
                    choices=list(PRESETS),
                    value=initial,
                    label="Model",
                    info="a released COCO checkpoint, or type/upload a path to your own "
                    "(.pt / .engine / .onnx / .xml) - task is read from it",
                    allow_custom_value=True,
                    scale=3,
                )
                model_size = gr.Textbox(
                    value=DEFAULT_INPUT_SIZE,
                    label="Input size",
                    info="PyTorch only - blank reads it from the checkpoint's config; "
                    "else `640` or `1024x2048`",
                    placeholder="auto",
                    scale=1,
                )
            with gr.Row():
                # OpenVINO needs its .bin sibling, which an upload drops - use the path box.
                model_file = gr.File(
                    label="…or upload weights (.pt / .onnx / .engine)",
                    file_types=[".pt", ".onnx", ".engine"],
                    type="filepath",
                    scale=1,
                )
                model_names = gr.Textbox(
                    label="Class names (optional)",
                    info="comma- or newline-separated, in class-id order; blank = the model's own",
                    placeholder="person, car, dog",
                    lines=3,
                    scale=2,
                )
            load_btn = gr.Button("Load model", variant="secondary")

        def load_and_collapse(*args):
            """Collapse the panel once loaded - but stay open on ❌ so the cause is in view."""
            status = load_backend(*args)
            return status, gr.Accordion(open=status.startswith("❌"))

        model_file.change(lambda p: p or "", inputs=model_file, outputs=model_spec)
        load_btn.click(  # one round-trip: status and panel state come back together
            fn=load_and_collapse,
            inputs=[model_spec, model_names, model_size],
            outputs=[model_status, model_panel],
        )

        with gr.Tabs():
            # ── Images: upload or webcam snapshot via bottom icons ───────
            with gr.TabItem("Images"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(
                            sources=["upload", "webcam"],
                            type="numpy",
                            label="Upload or Capture",
                        )
                        img_backend = gr.Radio(
                            ["D-FINE-seg", "SAM3"], value=DEFAULT_BACKEND, label="Backend"
                        )
                        img_prompt = gr.Textbox(
                            value="person",
                            label="Text prompt",
                            info="used only when Backend is SAM3",
                        )
                        img_conf_thresh = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=DEFAULT_CONF_THRESH,
                            label="Confidence threshold",
                        )
                        img_minimize = gr.Checkbox(
                            value=False,
                            label="Minimize visualization (boxes only, no labels)",
                        )
                        img_btn = gr.Button("Run", variant="primary")
                    with gr.Column():
                        img_out = gr.Image(type="numpy", label="Result", format="png")
                img_btn.click(
                    fn=predict_image,
                    inputs=[img_in, img_backend, img_prompt, img_conf_thresh, img_minimize],
                    outputs=img_out,
                )

            # ── Video: upload file ───────────────────────────────────────
            with gr.TabItem("Video"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(
                            sources=["upload"],
                            label="Upload Video",
                            format="mp4",
                        )
                        vid_backend = gr.Radio(
                            ["D-FINE-seg", "SAM3"], value=DEFAULT_BACKEND, label="Backend"
                        )
                        vid_prompt = gr.Textbox(
                            value="person",
                            label="Text prompt",
                            info="used only when Backend is SAM3",
                        )
                        vid_conf_thresh = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=DEFAULT_CONF_THRESH,
                            label="Confidence threshold",
                        )
                        vid_stride = gr.Slider(
                            minimum=1,
                            maximum=30,
                            step=1,
                            value=1,
                            label="Frame stride (1 = every frame)",
                        )
                        vid_minimize = gr.Checkbox(
                            value=False,
                            label="Minimize visualization (boxes only, no labels)",
                        )
                        vid_btn = gr.Button("Run", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="Annotated Video")
                vid_btn.click(
                    fn=predict_video,
                    inputs=[
                        vid_in,
                        vid_backend,
                        vid_prompt,
                        vid_conf_thresh,
                        vid_stride,
                        vid_minimize,
                    ],
                    outputs=vid_out,
                )

    return demo


def main(
    model: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    host: str = "0.0.0.0",  # LAN-reachable: the page loads arbitrary local paths as models
    port: int = 7860,
    share: bool = False,
) -> None:
    # gradio 6.16 trips this inside its own queue route, once per request
    warnings.filterwarnings("ignore", "'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated")
    if host not in ("127.0.0.1", "localhost"):
        # The Model panel loads any path the browser sends, so this hands local-file
        # probing and untrusted-graph deserialization to everyone who can reach the port.
        print(
            f"WARNING: serving on {host} - anyone who can reach this port can load "
            "any file on this machine as a model"
        )
    build_ui(model, task).launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
