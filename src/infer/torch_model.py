from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.d_fine.dfine import build_model
from src.d_fine.postprocess import DFINEPostProcessor
from src.dl.utils import filter_preds


class Torch_model:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_outputs: int,
        input_width: int = 640,
        input_height: int = 640,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        rect: bool = True,  # cuts paddings, inference is faster, accuracy might be lower
        half: bool = False,
        keep_ratio: bool = True,
        device: str = None,
    ):
        self.mean_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std_norm = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.input_size = (input_width, input_height)
        self.n_outputs = n_outputs
        self.model_name = model_name
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.debug_mode = False

        self.conf_thresh = conf_thresh

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.half:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

        self.postprocess = DFINEPostProcessor(num_classes=self.n_outputs, use_focal_loss=True)

        self._load_model()
        self._test_pred()

    def _load_model(self):
        self.model = build_model(self.model_name, self.n_outputs, self.device, None)
        self.model.load_state_dict(
            torch.load(self.model_path, weights_only=True, map_location=torch.device("cpu"))
        )

        if self.half:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)

    def _test_pred(self):
        random_image = np.random.randint(0, 255, size=(1000, 1110, 3), dtype=np.uint8)
        self.model(self._preprocess(random_image))

    def _compute_nearest_size(self, shape, target_size, stride=32) -> Tuple[int, int]:
        """
        Get nearest size that is divisible by 32
        """
        scale = target_size / max(shape)
        new_shape = [int(round(dim * scale)) for dim in shape]

        # Make sure new dimensions are divisible by the stride
        new_shape = [max(stride, int(np.ceil(dim / stride) * stride)) for dim in new_shape]
        return new_shape

    def _preprocess(self, img: NDArray, stride: int = 32) -> torch.tensor:
        if not self.keep_ratio:
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]), cv2.INTER_AREA)
        elif self.rect:
            target_height, target_width = self._compute_nearest_size(
                img.shape[:2], max(self.input_size[0], self.input_size[1])
            )
            img = letterbox(img, (target_height, target_width), stride=stride, auto=False)[0]
        else:
            img = letterbox(
                img, (self.input_size[1], self.input_size[0]), stride=stride, auto=False
            )[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        img = img.reshape([1, *img.shape])  # Add batch dimension

        # save debug image
        if self.debug_mode:
            debug_img = img[0].transpose(1, 2, 0)  # CHW to HWC
            debug_img = (debug_img * 255.0).astype(np.uint8)  # Convert to uint8
            debug_img = debug_img[:, :, ::-1]  # RGB to BGR for saving
            cv2.imwrite("torch_infer.jpg", debug_img)
        return torch.tensor(img).to(self.device)

    def _postprocess(self, preds: torch.tensor, target_shape, origin_h: int, origin_w: int):
        results = {}
        output = self.postprocess(
            preds, torch.tensor([[target_shape[1], target_shape[0]]], device=self.device)
        )
        output = filter_preds(output, self.conf_thresh)

        # only 1 batch
        boxes, scores, class_ids = (output[0]["boxes"], output[0]["scores"], output[0]["labels"])

        if self.keep_ratio:
            results["boxes"] = (
                scale_boxes_ratio_kept(target_shape, boxes, (origin_h, origin_w)).cpu().numpy()
            )
        else:
            results["boxes"] = scale_boxes(boxes, (origin_h, origin_w), target_shape).cpu().numpy()
        results["scores"] = scores.cpu().numpy()
        results["class_ids"] = class_ids.cpu().numpy()
        return results

    def _predict(self, img) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Returns predictions as a Tuple of torch.tensors
        - boxes (torch.tensor): Tensor of shape (N, 4) containing bounding boxes in [x1, y1, x2, y2] format.
        - scores (torch.tensor): Tensor of shape (N,) containing confidence scores for each box.
        - classes (torch.tensor): Tensor of shape (N,) containing class ids for each box.
        """
        return self.model(img)

    @torch.no_grad()
    def __call__(self, image: NDArray[np.uint8]):
        """
        Input image as ndarray, BGR, HWC
        """
        processed_image = self._preprocess(image)
        pred = self._predict(processed_image)
        return self._postprocess(pred, processed_image.shape[2:4], image.shape[0], image.shape[1])


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


# def non_max_suppression(boxes, scores, classes, score_threshold=0.5, iou_threshold=0.5):
#     """
#     Applies Non-Maximum Suppression (NMS) to filter bounding boxes.

#     Parameters:
#     - boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes in [x1, y1, x2, y2] format.
#     - scores (torch.Tensor): Tensor of shape (N,) containing confidence scores for each box.
#     - classes (torch.Tensor): Tensor of shape (N,) containing class indices for each box.
#     - score_threshold (float): Minimum confidence score to consider a box for NMS.
#     - iou_threshold (float): Intersection Over Union (IOU) threshold for NMS.

#     Returns:
#     - filtered_boxes (torch.Tensor): Tensor containing filtered bounding boxes after NMS.
#     - filtered_scores (torch.Tensor): Tensor containing confidence scores of the filtered boxes.
#     - filtered_classes (torch.Tensor): Tensor containing class indices of the filtered boxes.
#     """
#     # Step 1: Filter out boxes with confidence scores below the threshold
#     score_mask = scores >= score_threshold
#     boxes = boxes[score_mask]
#     scores = scores[score_mask]
#     classes = classes[score_mask]

#     # Prepare lists to collect the filtered boxes, scores, and classes
#     filtered_boxes = []
#     filtered_scores = []
#     filtered_classes = []

#     # Get unique classes present in the detections
#     unique_classes = classes.unique()

#     # Step 2: Perform NMS for each class separately
#     for unique_class in unique_classes:
#         # Get indices of boxes belonging to the current class
#         cls_mask = classes == unique_class
#         cls_boxes = boxes[cls_mask]
#         cls_scores = scores[cls_mask]

#         # Apply NMS for the current class
#         nms_indices = nms(cls_boxes, cls_scores, iou_threshold)

#         # Collect the filtered boxes, scores, and classes
#         filtered_boxes.append(cls_boxes[nms_indices])
#         filtered_scores.append(cls_scores[nms_indices])
#         filtered_classes.append(classes[cls_mask][nms_indices])

#     # Step 3: Concatenate the results
#     if filtered_boxes:
#         filtered_boxes = torch.cat(filtered_boxes)
#         filtered_scores = torch.cat(filtered_scores)
#         filtered_classes = torch.cat(filtered_classes)
#     else:
#         # If no boxes remain after NMS, return empty tensors
#         filtered_boxes = torch.empty((0, 4))
#         filtered_scores = torch.empty((0,))
#         filtered_classes = torch.empty((0,), dtype=classes.dtype)

#     return filtered_boxes, filtered_scores, filtered_classes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes_ratio_kept(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_boxes(boxes, orig_shape, resized_shape):
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes
