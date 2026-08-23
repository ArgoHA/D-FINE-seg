from typing import Dict, List, Union

import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from PIL import Image
from transformers import Sam3Model, Sam3Processor

MODEL_ID = "facebook/sam3"


class SAM3Model:
    """Text-promptable instance segmentation via SAM3, exposed with the same
    call signature ( __call__(img) -> [dict] ) as the D-FINE-seg wrappers.

    Multi-class: ``prompt`` takes ``"car, person"`` or ``["car", "person"]`` - each
    prompt runs one forward and becomes one class (``labels`` = prompt index).
    """

    def __init__(
        self,
        model_path: str = MODEL_ID,  # HF id or local path
        prompt: Union[str, List[str]] = "person",
        conf_thresh: float = 0.5,
        mask_threshold: float = 0.5,  # SAM3 always binarizes masks internally
        device: str = None,
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
        self.prompts = self.parse_prompts(prompt)
        self.conf_thresh = conf_thresh
        self.mask_threshold = mask_threshold

        self.processor, self.model = self._load(model_path)
        self.model = self.model.to(self.device).eval()
        logger.info(f"SAM3 model, Device: {self.device}, prompts: {self.prompts}")

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

    @torch.inference_mode()
    def __call__(
        self, img: NDArray[np.uint8], prompts: Union[str, List[str]] = None, bgr: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Input image as ndarray (BGR, HWC). Pass ``bgr=False`` for RGB input.
        ``prompts`` overrides ``self.prompts`` for this call (and persists).
        Output: list of length 1 with dict {"labels", "boxes", "scores", "masks"};
        one forward per prompt, all detections merged, ``labels`` = prompt index; a prompt
        that finds nothing contributes nothing (its masks are not at the image resolution).
        """
        if prompts is not None:
            self.prompts = self.parse_prompts(prompts)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img
        h, w = rgb.shape[:2]

        boxes, scores, masks, labels = [], [], [], []
        for cls_idx, text in enumerate(self.prompts):
            inputs = self.processor(images=Image.fromarray(rgb), text=text, return_tensors="pt").to(
                self.device
            )
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.conf_thresh,
                mask_threshold=self.mask_threshold,
                target_sizes=[(h, w)],
            )[0]
            if not len(res["scores"]):
                continue  # a prompt that found nothing keeps SAM3's mask resolution - never cat it
            boxes.append(res["boxes"].cpu().float())
            scores.append(res["scores"].cpu().float())
            masks.append(res["masks"].cpu().to(torch.uint8))  # already binary (0/1)
            labels.append(torch.full((len(res["boxes"]),), cls_idx, dtype=torch.long))
        if not boxes:
            return [
                {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "boxes": torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "masks": torch.zeros(0, h, w, dtype=torch.uint8),
                }
            ]
        return [
            {
                "labels": torch.cat(labels),
                "boxes": torch.cat(boxes),
                "scores": torch.cat(scores),
                "masks": torch.cat(masks),
            }
        ]
