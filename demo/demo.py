"""
D-FINE-seg Gradio Demo — Object Detection & Instance Segmentation

Backends selectable in the UI:
  D-FINE-seg — local checkpoint, format picked by file extension:
    .pt      -> PyTorch   (CUDA / MPS / CPU)
    .engine  -> TensorRT  (CUDA)
    .xml     -> OpenVINO  (CPU / iGPU)
  SAM3       — text-promptable instance segmentation (facebook/sam3, lazy-loaded)

Tabs:
  1. Images - upload or webcam snapshot -> annotated result
  2. Video  - upload a video file -> annotated output

Configure the variables below, then run:
python -m demo.demo
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

# ─── User configuration ─────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.resolve() / "model.pt"  # .pt | .engine | .xml
MODEL_NAME = "s"  # n / s / m / l / x  (only needed for .pt)
CLASSES = {0: "class_1", 1: "class_2"}
IM_WIDTH = 640  # only for .pt; auto-detected for .engine / .xml
IM_HEIGHT = 640
DEFAULT_CONF_THRESH = 0.5  # initial slider value
ENABLE_MASK_HEAD = False  # only for .pt; auto-detected for .engine / .xml
# ─────────────────────────────────────────────────────────────────────────


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


# ─── Helpers ─────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_path: str,
    model_name: str,
    classes: dict,
    im_width: int,
    im_height: int,
    conf_thresh: float,
    enable_mask_head: bool,
):
    ext = Path(model_path).suffix.lower()
    device = get_device()

    if ext == ".pt":
        from src.infer.torch_model import Torch_model

        return Torch_model(
            model_name=model_name,
            model_path=model_path,
            n_outputs=len(classes),
            input_width=im_width,
            input_height=im_height,
            conf_thresh=conf_thresh,
            enable_mask_head=enable_mask_head,
            device=device,
        )
    elif ext == ".engine":
        from src.infer.trt_model import TRT_model

        return TRT_model(
            model_path=model_path,
            n_outputs=len(classes),
            conf_thresh=conf_thresh,
            device=device,
        )
    elif ext == ".xml":
        from src.infer.ov_model import OV_model

        return OV_model(
            model_path=model_path,
            conf_thresh=conf_thresh,
        )
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .pt, .engine, or .xml")


# ─── SAM3 (text-promptable) backend ─────────────────────────────────────
SAM3_MODEL_ID = "facebook/sam3"

_sam_model = None


def _get_sam_model():
    """Lazy-load SAM3 on first use (heavy download)."""
    global _sam_model
    if _sam_model is None:
        from src.infer.sam3_model import SAM3_model

        print(f"Loading {SAM3_MODEL_ID} …")
        _sam_model = SAM3_model(model_path=SAM3_MODEL_ID, conf_thresh=DEFAULT_CONF_THRESH)
    return _sam_model


# ─── Initialization ─────────────────────────────────────────────────────
DEFAULT_BACKEND = "D-FINE-seg"
device = get_device()
model = load_model(
    MODEL_PATH, MODEL_NAME, CLASSES, IM_WIDTH, IM_HEIGHT, DEFAULT_CONF_THRESH, ENABLE_MASK_HEAD
)
visualizer = Visualizer(n_classes=len(CLASSES), class_names=CLASSES)
sam_visualizer = Visualizer(n_classes=1)  # single prompt class; name set per-run


# ─── Inference helpers ───────────────────────────────────────────────────
def _set_model_conf_threshold(conf_thresh: float) -> None:
    """Set a uniform confidence threshold for the currently loaded backend."""
    conf = float(np.clip(conf_thresh, 0.0, 1.0))
    if hasattr(model, "conf_threshs") and model.conf_threshs is not None:
        model.conf_threshs = [conf] * len(model.conf_threshs)
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
    _set_model_conf_threshold(conf_thresh)
    return model, visualizer


def _run_on_bgr(img_bgr, model_obj, vis_obj, minimize: bool = False) -> np.ndarray:
    """Run model + visualizer on a single BGR frame. Returns annotated BGR."""
    results = model_obj(img_bgr)
    return vis_obj.draw(img_bgr, results[0], minimize=minimize)


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
            frame = vis_obj.draw(frame, last_results, minimize=minimize)
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
def _toggle_prompt(backend: str):
    """Show the text-prompt box only for SAM3."""
    return gr.update(visible=(backend == "SAM3"))


model_info = (
    f"**D-FINE model:** `{Path(MODEL_PATH).name}` &ensp;|&ensp; "
    f"**Device:** `{device}` &ensp;|&ensp; "
    f"**Classes:** {len(CLASSES)} &ensp;|&ensp; "
    f"**SAM3:** `{SAM3_MODEL_ID}` (text-promptable)"
)

with gr.Blocks(title="D-FINE-seg + SAM3 Demo") as demo:
    gr.Markdown(f"# D-FINE-seg + SAM3 Demo\n{model_info}")

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
                        label="SAM3 text prompt",
                        visible=False,
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
            img_backend.change(_toggle_prompt, inputs=img_backend, outputs=img_prompt)
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
                        label="SAM3 text prompt",
                        visible=False,
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
            vid_backend.change(_toggle_prompt, inputs=vid_backend, outputs=vid_prompt)
            vid_btn.click(
                fn=predict_video,
                inputs=[vid_in, vid_backend, vid_prompt, vid_conf_thresh, vid_stride, vid_minimize],
                outputs=vid_out,
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
