"""
GPU-resident batched script: NVDEC -> batched TRT -> GPU paint -> NVENC.

This is an example script that fully utilizes a GPU. In this case we just
redact detected objects.
"""

import queue
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# Reuse the baseline's model loader + helpers (sets up the D-FINE sys.path on import).
from loguru import logger

GPU_ID = 0
FILL_VALUE = 114  # solid gray painted over each person
MASK_SCALE = 0.5  # compute/paint masks at this fraction of output res (coarser=faster); 1.0=full
DILATE_PX = 7  # mask dilation in full-res px (scaled into coarse-mask space); 0 disables
BATCH_WINDOW_S = 0.004  # how long the server waits to fill a batch before firing
MAX_CONCURRENT_CLIPS = 8  # == NVENC session cap on GeForce; also the max TRT batch
GPU_ENCODE_INPUT = True  # True: feed NVENC the device NV12 buffer; False: host-copy fallback


# --------------------------------------------------------------------------- #
# Frame buffers travel through the pipeline in CHW layout, not HWC. F.interpolate
# already produces CHW-contiguous output, so we skip the permute+contiguous copy
# that the old HWC path paid on every frame, and per-channel reads in
# rgb_to_nv12 / to_model_input become contiguous slabs instead of strided byte-3
# accesses. Decoder still yields HWC; resize_rgb is the one-time pivot.
# --------------------------------------------------------------------------- #


def autobackend(
    model_path: str,
    n_outputs: int,
    model_name: str,
    conf_thresh: float,
    enable_mask_head: bool,
    labels_to_use: list,
):
    if ".engine" in model_path:
        from src.infer.trt_model import TRT_model

        logger.info("TensorRT backend")
        return TRT_model(
            model_path=model_path,
            n_outputs=n_outputs,
            conf_thresh=conf_thresh,
            labels_to_use=labels_to_use,
        )

    elif ".pt" in model_path:
        import torch

        from src.infer.torch_model import Torch_model

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        logger.info(f"Torch backend, device: {device}")
        return Torch_model(
            model_path=model_path,
            n_outputs=n_outputs,
            model_name=model_name,
            conf_thresh=conf_thresh,
            enable_mask_head=enable_mask_head,
            device=device,
            labels_to_use=labels_to_use,
        )


def compute_out_size(src_w, src_h, max_dim):
    """
    Scale (src_w, src_h) down so the long edge is at most max_dim, preserving
    aspect ratio and rounding to even dimensions (required by most encoders).
    max_dim of None/0 (or a frame already within it) returns the source size.
    """
    if not max_dim or max(src_w, src_h) <= max_dim:
        return src_w, src_h
    scale = max_dim / max(src_w, src_h)
    out_w = max(2, round(src_w * scale / 2) * 2)
    out_h = max(2, round(src_h * scale / 2) * 2)
    return out_w, out_h


def probe_video(video):
    """Return (width, height, fps) of the first video stream via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()
    width, height = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    fps = float(num) / float(den)
    return width, height, fps


# Colorspace: RGB -> NV12 (BT.709, limited/"video" range) on the GPU.
# NVENC's universal input format is NV12 (Y plane HxW, then interleaved UV at
# half resolution). @torch.compile fuses the elementwise math so R/G/B/Y/U/V
# never materialize as full-res FP32 buffers — ~5x less DRAM traffic than the
# eager version. dynamic=True compiles once across varying H, W.
@torch.compile(dynamic=True)
def rgb_to_nv12(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: [3, H, W] uint8 CUDA (H, W even) -> NV12 [H*3//2, W] uint8 CUDA."""
    r = rgb[0].float()
    g = rgb[1].float()
    b = rgb[2].float()
    y = (0.1826 * r + 0.6142 * g + 0.0620 * b + 16.0).clamp(0, 255).to(torch.uint8)
    u = -0.1006 * r - 0.3386 * g + 0.4392 * b + 128.0
    v = 0.4392 * r - 0.3989 * g - 0.0403 * b + 128.0
    # 2x2-average chroma to half res in one kernel each, then stack+reshape to
    # interleave U,V across each row (U V U V ...) — the NV12 chroma layout.
    u2 = F.avg_pool2d(u[None, None], 2)[0, 0].clamp(0, 255).to(torch.uint8)
    v2 = F.avg_pool2d(v[None, None], 2)[0, 0].clamp(0, 255).to(torch.uint8)
    uv = torch.stack([u2, v2], dim=-1).reshape(u2.shape[0], -1)
    return torch.cat([y, uv], dim=0)


def resize_rgb(rgb: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """[H, W, 3] uint8 CUDA -> [3, out_h, out_w] uint8 CUDA (bilinear, CHW).

    Outputs CHW because everything downstream (paint, NV12, model preprocess)
    is happier in CHW. F.interpolate already produces CHW contiguous on the
    resize path, so we drop the permute+contiguous copy the HWC path paid.
    Pass-through still has to permute, but that case doesn't fire when we
    downscale (the common case).
    """
    if rgb.shape[0] == out_h and rgb.shape[1] == out_w:
        return rgb.permute(2, 0, 1).contiguous()
    x = rgb.permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.squeeze(0).clamp_(0, 255).to(torch.uint8)


def paint_gpu(frame: torch.Tensor, result: dict, fill: int = FILL_VALUE) -> None:
    """Solid-fill every person region into `frame` ([3, H, W] uint8 CUDA) in place.

    Mask-head engines: masks arrive at the coarse MASK_SCALE resolution; we
    union + dilate them there, then nearest-upsample the single union to the
    full frame once. Detection-only engines: fall back to painting each bbox
    rectangle (boxes come back already rescaled to (H, W) — see process_clip).
    """
    h, w = frame.shape[1], frame.shape[2]
    masks = result.get("masks")
    if masks is not None and masks.shape[0] > 0:
        combined = masks.any(dim=0).float()[None, None]  # [1, 1, mh, mw] coarse
        if DILATE_PX:
            k = max(1, round(DILATE_PX * MASK_SCALE)) | 1  # dilate in coarse-mask px, odd kernel
            combined = F.max_pool2d(combined, k, stride=1, padding=k // 2)
        up = F.interpolate(combined, size=(h, w), mode="nearest")[0, 0] > 0
        frame[:, up] = fill
        return

    boxes = result.get("boxes")
    if boxes is None or boxes.shape[0] == 0:
        return
    if DILATE_PX:
        pad = torch.tensor(
            [-DILATE_PX, -DILATE_PX, DILATE_PX, DILATE_PX],
            device=boxes.device,
            dtype=boxes.dtype,
        )
        boxes = boxes + pad
    x1 = boxes[:, 0].clamp(0, w).long()[:, None, None]
    y1 = boxes[:, 1].clamp(0, h).long()[:, None, None]
    x2 = boxes[:, 2].clamp(0, w).long()[:, None, None]
    y2 = boxes[:, 3].clamp(0, h).long()[:, None, None]
    ys = torch.arange(h, device=boxes.device)[None, :, None]
    xs = torch.arange(w, device=boxes.device)[None, None, :]
    inside = ((xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)).any(dim=0)
    frame[:, inside] = fill


# --------------------------------------------------------------------------- #
# Single shared batched-inference server. Workers submit one frame and block on
# the Future; the server coalesces concurrent submissions into one engine call.
# --------------------------------------------------------------------------- #
class BatchInferenceServer:
    def __init__(self, model, max_batch: int, window_s: float = BATCH_WINDOW_S):
        self.model = model
        self.max_batch = max_batch
        self.window_s = window_s
        self._q: queue.Queue = queue.Queue()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def infer(
        self,
        frame: torch.Tensor,
        original_size: tuple[int, int],
        input_ready: torch.cuda.Event,
    ) -> tuple[dict, torch.cuda.Event]:
        """Blocking submit. `frame` is the output-resolution uint8 RGB CHW tensor
        the worker just resized — gpu_run owns the engine-input resize / cast /
        /255 internally. `input_ready` is recorded on the caller's stream after
        `frame` was written; the engine stream waits on it before reading.
        `original_size` is (H, W) for postprocess (boxes rescaled + masks resized
        into it) — pass (mask_h, mask_w) to get coarse masks cheaply.

        Returns (result_dict, masks_ready). `masks_ready` is recorded on the
        engine stream after postprocess; callers `wait_event` it on their own
        stream before reading masks instead of paying a CPU sync.
        """
        fut: Future = Future()
        self._q.put((frame, original_size, input_ready, fut))
        return fut.result()

    @property
    def has_masks(self) -> bool:
        return self.model.has_masks

    def stop(self) -> None:
        self._stop = True
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop:
            try:
                first = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.window_s
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results, masks_ready = self._run_batch(
                    [b[0] for b in batch],
                    [b[1] for b in batch],
                    [b[2] for b in batch],
                )
                for (_, _, _, fut), res in zip(batch, results):
                    fut.set_result((res, masks_ready))
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter
                for *_, fut in batch:
                    fut.set_exception(exc)

    def _run_batch(self, frame_list, orig_sizes, input_events):
        # Hand the whole batch to the model's gpu_run, which does the engine-
        # input resize + cast + /255 + predict + postprocess on its private
        # stream and returns a masks_ready event for callers to wait on.
        # gpu_run only takes a single input_ready event, so the multi-worker
        # wait fans out here (one wait_event per submitter on model._stream).
        model = self.model
        for ev in input_events:
            model._stream.wait_event(ev)
        return model.gpu_run(frame_list, original_sizes=orig_sizes)


# --------------------------------------------------------------------------- #
# NVDEC / NVENC adapters — VERIFY against your installed PyNvVideoCodec version.
# --------------------------------------------------------------------------- #
class GpuDecoder:
    """NVDEC -> RGB uint8 [H, W, 3] CUDA tensors."""

    def __init__(self, path: Path, gpu_id: int = GPU_ID):
        import PyNvVideoCodec as nvc  # noqa: N813  # VERIFY: import name

        # VERIFY: constructor + RGB output flag. If your version only emits NV12,
        # decode NV12 here and add an NV12->RGB step (inverse of rgb_to_nv12).
        self._dec = nvc.SimpleDecoder(
            str(path),
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )

    def __iter__(self):
        for frame in self._dec:  # VERIFY: iteration yields DLPack-capable frames
            yield torch.from_dlpack(frame)  # [H, W, 3] uint8 CUDA

    def close(self) -> None:
        # No portable SimpleDecoder.close() across versions; drop our reference so
        # refcounting frees the NVDEC session promptly even if a clip errored.
        self._dec = None


class _AppCAI:
    """Minimal CUDA-Array-Interface holder for one NV12 plane (no torch 'stream'
    field, so the encoder's CAI reader can't trip on it)."""

    def __init__(self, ptr: int, shape: tuple[int, ...]):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": "|u1",
            "data": (ptr, False),
            "strides": None,  # planes are C-contiguous views of one buffer
        }


class _AppFrame:
    """GPU input for NVENC (usecpuinputbuffer=False). The encoder calls .cuda()
    and reads __cuda_array_interface__ on each plane. NV12 = [luma (H,W,1),
    chroma (H/2,W/2,2)] views into one contiguous device buffer: NVENC wants the
    interleaved UV described as 2 channels at half width, and luma with an explicit
    trailing channel (it rejects 2-D [H,W]). Same bytes, just the plane shapes."""

    def __init__(self, nv12: torch.Tensor):
        self._nv12 = nv12.contiguous()  # keep the buffer alive for the Encode call
        h = self._nv12.shape[0] * 2 // 3
        w = self._nv12.shape[1]
        base = self._nv12.data_ptr()
        self._planes = [_AppCAI(base, (h, w, 1)), _AppCAI(base + h * w, (h // 2, w // 2, 2))]

    def cuda(self):
        return self._planes


class GpuEncoder:
    """NVENC (NV12 in) -> raw HEVC bitstream piped into an ffmpeg MP4 muxer."""

    _schema_logged = False  # one-shot log of the NVENC packet-dict shape

    def __init__(self, out_path: Path, width: int, height: int, fps: float, gpu_id: int = GPU_ID):
        import PyNvVideoCodec as nvc  # noqa: N813  # VERIFY: import name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # GPU_ENCODE_INPUT=True keeps the NV12 buffer on-device (fed via _AppFrame.cuda());
        # False uploads it from host. NVENC compute runs on the encode ASIC either way.
        # VERIFY: CreateEncoder signature / kwargs (codec, preset, bitrate, gop).
        self._enc = nvc.CreateEncoder(
            width,
            height,
            "NV12",
            usecpuinputbuffer=not GPU_ENCODE_INPUT,
            codec="hevc",
            preset="P4",
            tuning_info="high_quality",
        )
        # NVENC emits an elementary stream; ffmpeg muxes it into a real .mp4.
        # To keep source audio, add `-i {src} -map 0:v -map 1:a -c:a aac` here.
        self._mux = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "hevc",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-c",
                "copy",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
        )

    def encode(self, nv12: torch.Tensor, ready_event: torch.cuda.Event | None = None) -> None:
        if GPU_ENCODE_INPUT:
            # Wait only on the event recorded after rgb_to_nv12 on the caller's
            # worker stream, not the whole current stream. PyNvVideoCodec's
            # Encode API takes no CUDA stream, so this CPU-side wait is the
            # required ordering point; making it event-scoped (vs. a global
            # default-stream synchronize) is what isolates one worker from
            # another's pending GPU work.
            if ready_event is not None:
                ready_event.synchronize()
            else:
                torch.cuda.current_stream().synchronize()
            packets = self._enc.Encode(_AppFrame(nv12))
        else:
            packets = self._enc.Encode(nv12.cpu().numpy())  # host-copy fallback
        self._write(packets)

    def close(self) -> None:
        self._write(self._enc.EndEncode())  # flush remaining packets
        self._mux.stdin.close()
        self._mux.wait()

    def _write(self, packets) -> None:
        # PyNvVideoCodec 2.x Encode()/EndEncode() return a list of per-packet
        # dicts (bitstream + metadata); empty while NVENC is still buffering.
        if not packets:
            return
        data = self._extract(packets)
        if data:
            self._mux.stdin.write(data)

    @classmethod
    def _extract(cls, packets) -> bytes:
        items = packets if isinstance(packets, (list, tuple)) else [packets]
        out = bytearray()
        for item in items:
            if isinstance(item, dict):
                if not cls._schema_logged:
                    cls._schema_logged = True
                    schema = {k: type(v).__name__ for k, v in item.items()}
                    logger.info(f"NVENC packet dict schema: {schema}")
                # the bitstream is the largest bytes-like value (metadata is tiny)
                bufs = [v for v in item.values() if isinstance(v, (bytes, bytearray, memoryview))]
                if bufs:
                    out += bytearray(max(bufs, key=len))
            else:  # already bytes-like / list of ints / buffer-protocol packet
                out += bytearray(item)
        return bytes(out)


# --------------------------------------------------------------------------- #
# Per-clip worker + pool
# --------------------------------------------------------------------------- #
def out_path_for(video_path: Path, data_path: str) -> Path:
    out_dir = Path(f"{data_path}_redacted") / video_path.parent.name
    return out_dir / f"{video_path.stem}_redacted{video_path.suffix}"


def process_clip(
    video_path: Path,
    server: BatchInferenceServer,
    data_path: str,
    max_dim,
    worker_stream: torch.cuda.Stream,
) -> int:
    src_w, src_h, fps = probe_video(video_path)
    out_w, out_h = compute_out_size(src_w, src_h, max_dim)
    # NV12 (rgb_to_nv12) needs even H/W; compute_out_size only guarantees that on
    # its downscale path, so clamp the pass-through case down to even.
    out_w, out_h = out_w & ~1, out_h & ~1
    out_path = out_path_for(video_path, data_path)
    decoder = GpuDecoder(video_path)
    encoder = GpuEncoder(out_path, out_w, out_h, fps)
    # Mask engines: ask for masks at coarse res — cuts per-instance upsample/cleanup.
    # Detection-only engines: ask for boxes in full frame space so paint_gpu can
    # rectangle-fill directly without rescaling.
    if server.has_masks:
        infer_h = max(1, round(out_h * MASK_SCALE))
        infer_w = max(1, round(out_w * MASK_SCALE))
    else:
        infer_h, infer_w = out_h, out_w

    n = 0
    try:
        for rgb_src in decoder:  # [H, W, 3] uint8 CUDA at source resolution
            with torch.cuda.stream(worker_stream):
                frame = resize_rgb(rgb_src, out_w, out_h)
                input_ready = torch.cuda.Event()
                input_ready.record(worker_stream)
                # gpu_run (inside server) handles the engine-input resize + cast
                # + /255; we just hand it the output-res uint8 frame. server.infer
                # blocks the Python thread until the engine picks up the batch,
                # but no CUDA ops are issued during the block.
                result, masks_ready = server.infer(frame, (infer_h, infer_w), input_ready)
                worker_stream.wait_event(masks_ready)
                paint_gpu(frame, result)
                nv12 = rgb_to_nv12(frame)
                encode_ready = torch.cuda.Event()
                encode_ready.record(worker_stream)
            encoder.encode(nv12, encode_ready)
            n += 1
    finally:
        encoder.close()
        decoder.close()
    logger.info(f"Saved redacted video: {out_path} ({n} frames)")
    return n


def worker(clip_q: queue.Queue, server, data_path, max_dim, totals: list) -> None:
    """Drain clips until the queue empties, then record this worker's frame total.

    Each worker owns one CUDA stream reused across clips so paint / NV12 / encode
    prep don't serialize on the default stream against the other 7 workers.
    `list.append` is atomic under the GIL, so collecting per-worker totals needs
    no lock — main sums them after every thread joins.
    """
    worker_stream = torch.cuda.Stream(device=GPU_ID)
    local = 0
    while True:
        try:
            video_path = clip_q.get_nowait()
        except queue.Empty:
            break
        try:
            local += process_clip(video_path, server, data_path, max_dim, worker_stream)
        except Exception:
            logger.exception(f"Clip failed: {video_path}")
    totals.append(local)


def main():
    model_path = "../../../nodes/D-FINE-seg/weights/coco_seg_batched.engine"
    data_path = "test_videos"
    out_max_dim = 1920  # downscale output so the long edge <= this (None/0 = source 4K)
    model_args = {
        "n_outputs": 80,
        "model_name": "s",
        "conf_thresh": 0.4,
        "enable_mask_head": True,
        "labels_to_use": [0],
    }

    # All clips across all cameras go into one queue; the pool drains it.
    cam_dirs = sorted(d for d in Path(data_path).iterdir() if d.is_dir())
    clip_q: queue.Queue = queue.Queue()
    for cam_dir in cam_dirs:
        for clip in sorted(p for p in cam_dir.iterdir() if p.suffix.lower() == ".mp4"):
            clip_q.put(clip)
    n_clips = clip_q.qsize()
    width = min(MAX_CONCURRENT_CLIPS, n_clips)
    logger.info(f"{n_clips} clips across {len(cam_dirs)} cameras; pool width {width}")

    # One engine, batched across the pool. Must be exported with a dynamic batch
    # profile whose max >= width.
    model = autobackend(model_path, **model_args)
    if not model.has_masks:
        logger.warning("engine has no mask head — falling back to bbox-rectangle redaction")
    server = BatchInferenceServer(model, max_batch=width)

    t0 = time.perf_counter()
    totals: list[int] = []
    args = (clip_q, server, data_path, out_max_dim, totals)
    threads = [threading.Thread(target=worker, args=args) for _ in range(width)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    server.stop()
    elapsed = time.perf_counter() - t0

    total_frames = sum(totals)
    fps = total_frames / elapsed if elapsed else 0.0
    logger.info(
        f"Processed {total_frames} frames from {n_clips} clips in {elapsed:.1f}s "
        f"({fps:.1f} frames/s aggregate)"
    )


if __name__ == "__main__":
    main()
