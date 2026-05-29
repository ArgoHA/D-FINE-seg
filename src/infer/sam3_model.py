from typing import Dict, List

import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from PIL import Image
from transformers import Sam3Model, Sam3Processor

MODEL_ID = "facebook/sam3"


class SAM3_model:
    """Text-promptable instance segmentation via SAM3, exposed with the same
    call signature ( __call__(img) -> [dict] ) as the D-FINE-seg wrappers."""

    def __init__(
        self,
        model_path: str = MODEL_ID,  # HF id or local path
        prompt: str = "person",
        conf_thresh: float = 0.5,
        mask_threshold: float = 0.5,  # SAM3 always binarizes masks internally
        device: str = None,
    ):
        # SAM3 autocasts to bf16; restrict to cuda/cpu (mps bf16 is unreliable)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        self.conf_thresh = conf_thresh
        self.mask_threshold = mask_threshold

        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path, dtype=torch.bfloat16)
        self.model = self.model.to(self.device).eval()
        logger.info(f"SAM3 model, Device: {self.device}")

    @torch.inference_mode()
    def __call__(
        self, img: NDArray[np.uint8], prompt: str = None, bgr: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Input image as ndarray (BGR, HWC). Pass ``bgr=False`` for RGB input.
        ``prompt`` overrides ``self.prompt`` for this call (and persists).
        Output: list of length 1 with dict {"labels", "boxes", "scores", "masks"};
        single prompt -> all labels are 0.
        """
        if prompt is not None:
            self.prompt = prompt
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img
        h, w = rgb.shape[:2]

        inputs = self.processor(
            images=Image.fromarray(rgb), text=self.prompt, return_tensors="pt"
        ).to(self.device)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.conf_thresh,
            mask_threshold=self.mask_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = res["boxes"].cpu().float()
        return [
            {
                "labels": torch.zeros(len(boxes), dtype=torch.long),
                "boxes": boxes,
                "scores": res["scores"].cpu().float(),
                "masks": res["masks"].cpu().to(torch.uint8),  # already binary (0/1)
            }
        ]
