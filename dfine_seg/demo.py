"""
D-FINE-seg Gradio Demo — detection, instance segmentation, semantic segmentation

Just run it — COCO detection weights download on first use:
    dfine-seg demo          (or: python -m dfine_seg.demo)

Everything is set from the UI; nothing here needs editing. The "Model" panel swaps in
your own checkpoint at runtime (size preset or a path/upload) and lets you name its
classes, so a freshly trained model can be tried on your images and videos immediately.

Backends selectable in the UI:
  D-FINE-seg — size preset (n|s|m|l|x) or a local artifact, format picked by extension:
    .pt      -> PyTorch   (CUDA / MPS / CPU)
    .engine  -> TensorRT  (CUDA)
    .onnx    -> ONNXRuntime
    .xml     -> OpenVINO  (CPU / iGPU)
  SAM3       — text-promptable instance segmentation (facebook/sam3, lazy-loaded)

Tabs:
  1. Images - upload or webcam snapshot -> annotated result
  2. Video  - upload a video file -> annotated output
"""

import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

from dfine_seg import load_model
from dfine_seg.loader import SIZES

# ─── Startup defaults (all overridable in the UI) ───────────────────────
DEFAULT_MODEL = "s"  # size (n|s|m|l|x) -> COCO weights, or a path to .pt/.engine/.onnx/.xml
DEFAULT_TASK = "auto"  # auto | detect | segment | sem_seg
DEFAULT_INPUT_SIZE = 640  # .pt only; graph artifacts carry their own input size
DEFAULT_CONF_THRESH = 0.5  # initial slider value
# ─────────────────────────────────────────────────────────────────────────

# The 10 released COCO checkpoints, label -> load_model(size, task). Labels mirror the
# Python call, and anything not in here is treated as a path, so one field covers both.
PRESETS = {f"{s} ({t})": (s, t) for t in ("detect", "segment") for s in SIZES}


class Visualizer:
    """Draws detection / segmentation results with consistent per-class colors."""

    def __init__(self, n_classes: int, class_names: Optional[Dict[int, str]] = None):
        self.class_names = class_names or {i: str(i) for i in range(n_classes)}
        self.colors = self._generate_colors(n_classes)

    @staticmethod
    def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
        """Evenly spaced hues on the HSV wheel → BGR tuples."""
        colors = []
        n = max(n, 1)
        for i in range(n):
            hue = int(180 * i / n)
            hsv = np.array([[[hue, 210, 210]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))
        return colors

    # ── public API ──────────────────────────────────────────────────────
    def draw(
        self, img: np.ndarray, results: Dict[str, torch.Tensor], minimize: bool = False
    ) -> np.ndarray:
        img = img.copy()
        labels = results["labels"]
        boxes = results["boxes"]
        scores = results["scores"]
        has_masks = "masks" in results and results["masks"] is not None

        if len(labels) == 0:
            return img

        # Adaptive sizes based on image resolution
        ref = max(img.shape[:2])
        box_thick = max(1, int(ref / 400))
        font_scale = max(0.35, ref / 1800)
        font_thick = max(1, int(ref / 600))
        edge_thick = max(1, int(ref / 350))

        # Masks first (underneath boxes)
        if has_masks:
            masks = results["masks"]
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            for i in range(len(labels)):
                label_id = int(labels[i].item())
                color = self.colors[label_id % len(self.colors)]
                self._draw_mask(img, masks[i], color, edge_thickness=edge_thick)

        # Boxes + labels
        for i in range(len(labels)):
            label_id = int(labels[i].item())
            score = float(scores[i].item())
            color = self.colors[label_id % len(self.colors)]
            name = self.class_names.get(label_id, str(label_id))
            x1, y1, x2, y2 = map(int, boxes[i].tolist())

            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)

            if not minimize:
                text = f"{name} {score:.2f}"
                self._draw_label(img, text, x1, y1, color, font_scale, font_thick)

        return img

    # ── private helpers ─────────────────────────────────────────────────
    @staticmethod
    def _draw_label(
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color: Tuple[int, int, int],
        font_scale: float,
        font_thick: int,
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
        pad = 4

        # Try placing above the box; fall back to below
        if y - th - 2 * pad >= 0:
            bg_y1, bg_y2, text_y = y - th - 2 * pad, y, y - pad
        else:
            bg_y1, bg_y2, text_y = y, y + th + 2 * pad, y + th + pad

        cv2.rectangle(img, (x, bg_y1), (x + tw + 2 * pad, bg_y2), bg_color, -1)

        # White or black text depending on background brightness (perceived luminance)
        lum = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
        txt_col = (0, 0, 0) if lum > 140 else (255, 255, 255)
        cv2.putText(img, text, (x + pad, text_y), font, font_scale, txt_col, font_thick)

    @staticmethod
    def _draw_mask(
        img: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int],
        body_alpha: float = 0.25,
        edge_alpha: float = 0.70,
        edge_thickness: int = 2,
    ):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        if mask.max() == 0:
            return

        # Semi-transparent body fill
        m = mask.astype(bool)
        overlay = np.full_like(img, color, dtype=np.uint8)
        img[m] = cv2.addWeighted(img[m], 1 - body_alpha, overlay[m], body_alpha, 0)

        # More opaque edge
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            edge_mask = np.zeros_like(mask)
            cv2.drawContours(edge_mask, contours, -1, 1, edge_thickness)
            e = edge_mask.astype(bool)
            edge_ov = np.full_like(img, color, dtype=np.uint8)
            img[e] = cv2.addWeighted(img[e], 1 - edge_alpha, edge_ov[e], edge_alpha, 0)


# ─── sem_seg rendering ───────────────────────────────────────────────────
@lru_cache(maxsize=8)
def _palette(n_classes: int) -> np.ndarray:
    """[256, 3] BGR lookup table; ids past the class count (incl. ignore=255) stay black."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:n_classes] = np.array(Visualizer._generate_colors(n_classes), dtype=np.uint8)
    return lut


def overlay_sem_seg(img: np.ndarray, label_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colorized dense label map over a BGR frame."""
    n = max(int(label_map.max()) + 1, 1)
    if label_map.shape != img.shape[:2]:
        label_map = cv2.resize(
            label_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return cv2.addWeighted(img, 1 - alpha, _palette(n)[label_map], alpha, 0)


# ─── Model loading (driven by the UI) ────────────────────────────────────
@dataclass
class Loaded:
    """The D-FINE backend currently serving both tabs."""

    model: object = None
    vis: Optional[Visualizer] = None
    names: Dict[int, str] = field(default_factory=dict)


CURRENT = Loaded()

# Backends whose class count isn't recoverable from the artifact: the postprocessor is
# fused, so nothing in the graph carries it (.pt reads the head, OpenVINO reads the graph).
OPAQUE_CLASS_COUNT = (".onnx", ".engine", ".mlpackage", ".tflite")


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


def load_backend(spec: str, names_text: str, input_size: float, task: str = "auto") -> str:
    """(Re)load the D-FINE backend from the UI controls; returns a status line."""
    src = (spec or "").strip() or DEFAULT_MODEL
    src, preset_task = PRESETS.get(src, (src, None))  # a preset carries its own task
    task = preset_task or task
    suffix = Path(src).suffix.lower()
    given = parse_names(names_text)

    kwargs = {"conf_thresh": DEFAULT_CONF_THRESH}
    if suffix == ".pt" and input_size:  # graph artifacts read it off the graph
        kwargs["input_height"] = kwargs["input_width"] = int(input_size)
    if suffix in OPAQUE_CLASS_COUNT:
        # These wrappers require a class count they can't read off their graph, and use it
        # only to size the per-class threshold list that labels index into. Never derive it
        # from the names box: naming 3 classes of an 8-class model would index out of
        # bounds mid-postprocess (a CUDA-side assert, i.e. a dead process). Overshooting is
        # free, so bound it above any realistic label space instead.
        kwargs["n_outputs"] = 4096
    # task selects the weights for a size preset and the architecture for a .pt; graph
    # artifacts have it baked in, and their wrappers take no task=.
    picked = None if task == "auto" or suffix not in ("", ".pt") else task

    try:
        model = load_model(src, task=picked, names=given, **kwargs)
    except Exception as e:  # keep the working model rather than leaving the demo dead
        kept = f" — keeping {Path(CURRENT.model.model_path).name}" if CURRENT.model else ""
        return f"❌ {type(e).__name__}: {e}{kept}"

    names = model.names or {}
    # What we actually know: the wrapper's own count, except where we just made it up above.
    known = 0 if suffix in OPAQUE_CLASS_COUNT else getattr(model, "n_outputs", 0) or 0
    known = max(known, max(names) + 1 if names else 0)
    CURRENT.model = model
    CURRENT.names = names
    CURRENT.vis = Visualizer(n_classes=known or 80, class_names=names or None)

    h, w = getattr(model, "input_size", (None, None))
    note = f" ({len(names)} named)" if 0 < len(names) < known else ("" if names else " (unnamed)")
    # Plain text: this line is both the UI status and the console log, so no markup.
    return (
        f"✅ {type(model).__name__} | {Path(src).name} | "
        f"task: {getattr(model, 'task', 'from graph')} | "
        f"classes: {known or '? — name them above'}{note if known else ''} | "
        f"device: {getattr(model, 'device', '?')} | input: {h}x{w}"
    )


# ─── SAM3 (text-promptable) backend ─────────────────────────────────────
SAM3_MODEL_ID = "facebook/sam3"

_sam_model = None


def _get_sam_model():
    """Lazy-load SAM3 on first use — seconds from the HF cache, a ~6.5 GB download without."""
    global _sam_model
    if _sam_model is None:
        from dfine_seg.infer.sam3_model import SAM3Model

        gr.Info(f"Loading {SAM3_MODEL_ID} — downloads ~6.5 GB if it isn't cached yet")
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
        raise gr.Error("No model loaded — fix the model settings above and press Load.")
    _set_model_conf_threshold(CURRENT.model, conf_thresh)
    return CURRENT.model, CURRENT.vis


def _run_on_bgr(img_bgr, model_obj, vis_obj, minimize: bool = False) -> np.ndarray:
    """Run model + visualizer on a single BGR frame. Returns annotated BGR."""
    return _draw(img_bgr, model_obj(img_bgr)[0], vis_obj, minimize=minimize)


def _draw(img_bgr, results: dict, vis_obj, minimize: bool = False) -> np.ndarray:
    """Boxes/masks, or a palette overlay when the model is dense (sem_seg)."""
    if "sem_seg" in results:
        return overlay_sem_seg(img_bgr, results["sem_seg"].cpu().numpy())
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
    print(f"[video] done — {idx} frames in {elapsed:.1f}s ({idx / elapsed:.1f} fps)")

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
                    "(.pt / .engine / .onnx / .xml) — task is read from it",
                    allow_custom_value=True,
                    scale=3,
                )
                model_size = gr.Number(
                    value=DEFAULT_INPUT_SIZE,
                    precision=0,
                    label="Input size",
                    info=".pt only — must match training",
                    scale=1,
                )
            with gr.Row():
                # OpenVINO needs its .bin sibling, which an upload drops — use the path box.
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
            """Collapse the panel once loaded — but stay open on ❌ so the cause is in view."""
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
    host: str = "0.0.0.0",  # reachable from another machine — the lab-box case
    port: int = 7860,
    share: bool = False,
) -> None:
    # gradio 6.16 trips this inside its own queue route, once per request
    warnings.filterwarnings("ignore", "'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated")
    build_ui(model, task).launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
