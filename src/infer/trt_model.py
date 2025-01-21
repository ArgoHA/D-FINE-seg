from typing import List, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch
from numpy.typing import NDArray

from src.d_fine.postprocess import DFINEPostProcessor
from src.dl.utils import filter_preds


class TRT_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        input_width: int = 640,
        input_height: int = 640,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        rect: bool = False,  # No need for rectangular inference with fixed size
        half: bool = False,
        keep_ratio: bool = True,
        device: str = None,
    ) -> None:
        self.input_size = (input_width, input_height)
        self.n_outputs = n_outputs
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.rect = rect
        self.half = half
        self.keep_ratio = keep_ratio
        self.debug_mode = True
        self.conf_thresh = conf_thresh
        self.d_fine_postprocessor = DFINEPostProcessor(num_classes=n_outputs)

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.half:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

        self._load_model()
        self._test_pred()

    def _load_model(self):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    @staticmethod
    def _torch_dtype_from_trt(trt_dtype):
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int8:
            return torch.int8
        else:
            raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")

    def _preprocess(self, img: NDArray, stride: int = 32) -> torch.Tensor:
        if not self.keep_ratio:
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]), cv2.INTER_AREA)
        else:
            img = letterbox(
                img, (self.input_size[1], self.input_size[0]), stride=stride, auto=False
            )[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img /= 255.0
        img = img.reshape([1, *img.shape])  # Add batch dimension

        # Save debug image if needed
        if self.debug_mode:
            debug_img = img[0].transpose(1, 2, 0)  # CHW to HWC
            debug_img = (debug_img * 255.0).astype(np.uint8)  # Convert to uint8
            debug_img = debug_img[:, :, ::-1]  # RGB to BGR for saving
            cv2.imwrite("trt_infer.jpg", debug_img)
        return torch.from_numpy(img).to(self.device)

    def _predict(self, img: torch.Tensor) -> List[torch.Tensor]:
        # Ensure the input tensor is contiguous
        img_tensor = img.contiguous()

        # Prepare bindings
        bindings = [None] * self.engine.num_bindings
        output_tensors = []

        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            binding_shape = self.engine.get_binding_shape(idx)
            binding_shape = tuple(binding_shape)  # Convert to tuple of integers
            binding_dtype = self.engine.get_binding_dtype(idx)
            torch_dtype = self._torch_dtype_from_trt(binding_dtype)

            if self.engine.binding_is_input(binding):
                # Input binding
                assert tuple(binding_shape) == tuple(
                    img_tensor.shape
                ), f"Input shape mismatch: expected {binding_shape}, got {img_tensor.shape}"
                bindings[idx] = int(img_tensor.data_ptr())
            else:
                # Output binding
                output_tensor = torch.empty(binding_shape, dtype=torch_dtype, device=self.device)
                output_tensors.append(output_tensor)
                bindings[idx] = int(output_tensor.data_ptr())

        # Run inference
        self.context.execute_v2(bindings)

        # Outputs are already on the device as PyTorch tensors
        return output_tensors

    def _test_pred(self) -> None:
        random_image = np.random.randint(
            0, 255, size=(self.input_size[1], self.input_size[0], 3), dtype=np.uint8
        )
        processed_image = self._preprocess(random_image)
        preds = self._predict(processed_image)
        self._postprocess(
            preds,
            processed_image.shape[2:4],
            random_image.shape[0],
            random_image.shape[1],
        )

    def _postprocess(
        self, preds: List[np.ndarray], target_shape: Tuple[int, int], origin_h: int, origin_w: int
    ):
        output = self.d_fine_postprocessor(
            {"pred_logits": preds[0], "pred_boxes": preds[1]},
            torch.tensor([[target_shape[1], target_shape[0]]], device=self.device),
        )
        output = filter_preds(output, self.conf_thresh)

        res = {"boxes": [], "scores": [], "class_ids": []}

        # only 1 batch
        if self.keep_ratio:
            res["boxes"] = (
                scale_boxes_ratio_kept(target_shape, output[0]["boxes"], (origin_h, origin_w))
                .cpu()
                .numpy()
            )
        else:
            res["boxes"] = (
                scale_boxes(output[0]["boxes"], (origin_h, origin_w), target_shape).cpu().numpy()
            )
        res["scores"] = output[0]["scores"].cpu().numpy()
        res["class_ids"] = output[0]["labels"].cpu().numpy()
        return res

    def __call__(self, image: NDArray[np.uint8]):
        processed_image = self._preprocess(image)
        pred = self._predict(processed_image)
        res = self._postprocess(pred, processed_image.shape[2:4], image.shape[0], image.shape[1])
        return res


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
