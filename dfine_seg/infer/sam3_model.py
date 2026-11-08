import math
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.typing import NDArray
from PIL import Image
from transformers import Sam3Model, Sam3Processor

MODEL_ID = "facebook/sam3"


class SAM3Model:
    """Text-promptable segmentation via SAM3, exposed with the same call signature
    ( __call__(img) -> [dict] ) as the D-FINE-seg wrappers.

    Multi-class: ``prompt`` takes ``"car, person"`` or ``["car", "person"]`` - each prompt
    becomes one class (``labels`` = prompt index) - or ``{"vehicle": ["car", "truck"], ...}``
    to merge several prompts into one class (``labels`` = class index, in dict order).

    The image is encoded once and all prompts share it, run as batches of ``prompt_batch``
    (text embeddings are cached). Matches one forward per prompt up to bf16 noise, ~8x faster
    at 20 prompts. ``compile=True`` adds torch.compile (~1.6x more, ~1 min warmup).

    ``task``:
    - ``segment``: instances with ``labels``, ``boxes``, ``scores``, ``masks``.
    - ``detect``: the same without ``masks``. Worth it when they are not used: SAM3 binarizes
      every candidate above ``conf_thresh`` at the original resolution, which at a low
      threshold on a 4K frame costs seconds and gigabytes. Boxes and scores are unaffected.
    - ``sem_seg``: ``sem_seg`` (H, W) class ids and ``scores`` (H, W). A pixel's class score is
      the max over the class's prompts and queries of ``query score * mask probability``, so
      every pixel gets a class; threshold ``scores`` to mark uncertain pixels as ignore.
      Measured on GOOSE 2048x1000: instance scores beat SAM3's own semantic head by ~0.06 mIoU.

    Like the official SAM3 inference, the whole image is resized to SAM3's square input.
    ``tiles > 1`` (sem_seg only, opt-in) instead splits it along the long side into that many
    tiles, each resized to the square; on GOOSE 2048x1000, 2 tiles gave ~+0.06 mIoU.
    """

    def __init__(
        self,
        model_path: str = MODEL_ID,  # HF id or local path
        prompt: Union[str, List[str], Dict[str, List[str]]] = "person",
        conf_thresh: float = 0.5,
        mask_threshold: float = 0.5,  # SAM3 always binarizes masks internally
        device: str = None,
        task: str = "segment",  # segment | detect | sem_seg
        prompt_batch: int = 20,  # prompts per forward; ~11 GB peak at 20 on a 16 GB GPU
        tiles: int = 1,  # sem_seg only: >1 splits the long side, not official SAM3 inference
        compile: bool = False,
    ):
        # Same fallback chain as the D-FINE wrappers: cuda -> mps -> cpu.
        # SAM3 autocasts to bf16, which mps handles (verified torch 2.13, macOS 26).
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.prompts = prompt
        self.conf_thresh = conf_thresh
        self.mask_threshold = mask_threshold
        self.task = task
        self.prompt_batch = prompt_batch
        self.tiles = tiles

        self.processor, self.model = self._load(model_path)
        self.model = self.model.to(self.device).eval()
        if compile:
            # Static shapes: the processor resizes every image to SAM3's square input and
            # _forward pads the last prompt batch, so each module compiles once.
            for name in ("vision_encoder", "detr_encoder", "detr_decoder", "mask_decoder"):
                module = getattr(self.model, name)
                compiled = torch.compile(module, mode="max-autotune-no-cudagraphs", dynamic=False)
                setattr(self.model, name, compiled)
        logger.info(f"SAM3 model, Device: {self.device}, task: {task}, classes: {self.class_names}")

    @property
    def prompts(self) -> List[str]:
        return self._prompts

    @prompts.setter
    def prompts(self, prompt) -> None:
        self.class_names, self._prompts, self.prompt_class = self.parse_classes(prompt)
        self._text = None  # text embeddings are recomputed for the new prompts

    @staticmethod
    def parse_prompts(prompt):
        """'' | 'person' | 'car, person' | ['car', 'person'] -> non-empty list of prompts.

        Commas and newlines split inside list items too, so both call styles agree.
        """
        if prompt is None:
            return ["object"]
        if isinstance(prompt, str):
            prompt = [prompt]
        parts = [
            part.strip()
            for value in prompt
            for line in str(value).splitlines()
            for part in line.split(",")
        ]
        return [part for part in parts if part] or ["object"]

    @classmethod
    def parse_classes(cls, prompt):
        """-> (class names, flat prompts, class index of each prompt).

        A dict maps class name -> prompts (same syntax as ``parse_prompts``); anything else
        makes every prompt its own class.
        """
        if not isinstance(prompt, dict):
            prompts = cls.parse_prompts(prompt)
            return list(prompts), prompts, list(range(len(prompts)))
        if not prompt:
            raise ValueError("empty class -> prompts mapping")
        names, prompts, prompt_class = [], [], []
        for idx, (name, synonyms) in enumerate(prompt.items()):
            names.append(str(name))
            parsed = cls.parse_prompts(synonyms if synonyms else str(name))
            prompts += parsed
            prompt_class += [idx] * len(parsed)
        return names, prompts, prompt_class

    @staticmethod
    def _load(model_path: str):
        """Load from the HF cache first, hub second.

        A repo id is otherwise always resolved through the hub, so a gated one (`facebook/sam3`)
        demands a valid token on every run even when the snapshot is already on disk.
        """
        try:
            return (
                Sam3Processor.from_pretrained(model_path, local_files_only=True),
                Sam3Model.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True),
            )
        except OSError:
            logger.info(f"{model_path} not in the local HF cache, fetching from the hub")
            return (
                Sam3Processor.from_pretrained(model_path),
                Sam3Model.from_pretrained(model_path, dtype=torch.bfloat16),
            )

    def _autocast(self):
        return torch.autocast(device_type=self.device, dtype=torch.bfloat16)

    def _text_embeds(self):
        if self._text is None:
            text = self.processor(text=self._prompts, return_tensors="pt", padding=True)
            text = text.to(self.device)
            with self._autocast():
                embeds = self.model.get_text_features(
                    input_ids=text.input_ids, attention_mask=text.attention_mask
                ).pooler_output
            self._text = embeds, text.attention_mask
        return self._text

    def _forward(self, rgb: NDArray[np.uint8]):
        """Encode the image once, then yield (first prompt, n prompts, outputs) per batch.

        Batches are balanced and the last one is padded with repeats of the final prompt, so
        every forward has the same shape; callers keep only the first n outputs.
        """
        embeds, mask = self._text_embeds()
        total = len(self._prompts)
        size = math.ceil(total / math.ceil(total / self.prompt_batch))
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self._autocast():
            vision = self.model.get_vision_features(
                pixel_values=inputs.pixel_values.to(self.model.dtype)
            )
            vision = type(vision)(
                **{
                    k: tuple(x.expand(size, *x.shape[1:]) for x in v)
                    if isinstance(v, (tuple, list))
                    else v.expand(size, *v.shape[1:])
                    for k, v in vision.items()
                }
            )
            for start in range(0, total, size):
                idx = torch.arange(start, start + size, device=embeds.device).clamp(max=total - 1)
                out = self.model(
                    vision_embeds=vision, text_embeds=embeds[idx], attention_mask=mask[idx]
                )
                yield start, min(size, total - start), out

    @torch.inference_mode()
    def __call__(
        self,
        img: NDArray[np.uint8],
        prompts: Union[str, List[str], Dict[str, List[str]]] = None,
        bgr: bool = True,
        conf_thresh: float = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Input image as ndarray (BGR, HWC). Pass ``bgr=False`` for RGB input.
        ``prompts`` overrides ``self.prompts`` for this call (and persists).
        ``conf_thresh`` overrides ``self.conf_thresh`` for this call only (not for sem_seg).
        Output: list of length 1 with one dict, see the class docstring per ``task``. For
        instances, a prompt that finds nothing contributes nothing.
        """
        if prompts is not None:
            self.prompts = prompts
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img
        if self.task == "sem_seg":
            return [self._sem_seg(rgb)]
        conf_thresh = self.conf_thresh if conf_thresh is None else conf_thresh
        h, w = rgb.shape[:2]

        detect = self.task == "detect"
        boxes, scores, masks, labels = [], [], [], []
        for start, n, outputs in self._forward(rgb):
            sizes = [(h, w)] * len(outputs.pred_logits)
            if detect:
                results = self.processor.post_process_object_detection(
                    outputs, threshold=conf_thresh, target_sizes=sizes
                )
            else:
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=conf_thresh,
                    mask_threshold=self.mask_threshold,
                    target_sizes=sizes,
                )
            for offset, res in enumerate(results[:n]):
                if not len(res["scores"]):
                    continue  # a prompt that found nothing keeps SAM3's mask resolution
                cls_idx = self.prompt_class[start + offset]
                boxes.append(res["boxes"].cpu().float())
                scores.append(res["scores"].cpu().float())
                labels.append(torch.full((len(res["boxes"]),), cls_idx, dtype=torch.long))
                if not detect:
                    masks.append(res["masks"].cpu().to(torch.uint8))  # already binary (0/1)
        if not boxes:
            empty = {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
            }
            if not detect:
                empty["masks"] = torch.zeros(0, h, w, dtype=torch.uint8)
            return [empty]
        merged = {
            "labels": torch.cat(labels),
            "boxes": torch.cat(boxes),
            "scores": torch.cat(scores),
        }
        if not detect:
            merged["masks"] = torch.cat(masks)
        return [merged]

    def _class_scores(self, rgb: NDArray[np.uint8]) -> torch.Tensor:
        """(C, H, W) per-class scores for one (near-square) image."""
        acc = None
        for start, n, out in self._forward(rgb):
            presence = out.presence_logits[:n].float().sigmoid()  # (n, 1)
            query = out.pred_logits[:n].float().sigmoid() * presence  # (n, Q)
            prob = out.pred_masks[:n].float().sigmoid()  # (n, Q, h, w)
            per_prompt = (query[:, :, None, None] * prob).amax(1)  # (n, h, w)
            if acc is None:
                acc = per_prompt.new_zeros((len(self.class_names), *per_prompt.shape[1:]))
            for offset in range(n):
                c = self.prompt_class[start + offset]
                acc[c] = torch.maximum(acc[c], per_prompt[offset])
        size = rgb.shape[:2]
        return F.interpolate(acc[None], size=size, mode="bilinear", align_corners=False)[0]

    def _sem_seg(self, rgb: NDArray[np.uint8]) -> Dict[str, torch.Tensor]:
        if self.tiles <= 1:
            scores, labels = self._class_scores(rgb).max(0)
            return {"sem_seg": labels.cpu(), "scores": scores.cpu()}
        h, w = rgb.shape[:2]
        axis = 1 if w >= h else 0
        edges = np.linspace(0, rgb.shape[axis], self.tiles + 1).round().astype(int)
        parts = []
        for a, b in zip(edges[:-1], edges[1:]):
            tile = rgb[:, a:b] if axis == 1 else rgb[a:b]
            parts.append(self._class_scores(np.ascontiguousarray(tile)))
        scores, labels = torch.cat(parts, dim=axis + 1).max(0)
        return {"sem_seg": labels.cpu(), "scores": scores.cpu()}
