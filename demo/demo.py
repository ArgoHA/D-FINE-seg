"""
Place model weights in the same folder as this file, update variables and run:
python -m demo.demo
"""

import time
from pathlib import Path

import cv2
import gradio as gr

from src.infer.torch_model import Torch_model


def visualize(img, preds):
    for pred in preds["boxes"]:
        print(pred)
        cv2.rectangle(
            img,
            (int(pred[0]), int(pred[1])),
            (int(pred[2]), int(pred[3])),
            (255, 0, 0),
            2,
        )
    return img


def predict_image(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    results = model(img_bgr)
    print((time.perf_counter() - t0) * 1000, "ms", results)
    res_img = visualize(img, results[0])
    return res_img


model_name = "s"
classes = {0: "class_name"}
im_width = 640
im_height = 640
conf_thresh = 0.5
device = "cpu"

f_path = Path(__file__).parent

model = Torch_model(
    model_name=model_name,
    model_path=f_path / "model.pt",
    n_outputs=len(classes),
    input_width=im_width,
    input_height=im_height,
    conf_thresh=conf_thresh,
    device=device,
)

iface = gr.Interface(
    fn=predict_image,
    inputs=[gr.Image(type="numpy", label="Upload Image")],
    outputs=gr.Image(type="numpy", label="Result", format="png"),
    title="HW detection",
    description="Upload images for inference.",
)

if __name__ == "__main__":
    iface.launch()
