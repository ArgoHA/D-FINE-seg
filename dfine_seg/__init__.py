"""D-FINE-seg: detection + instance segmentation + semantic segmentation.

```python
from dfine_seg import load_model, read_image, Visualizer

model = load_model("s")                       # COCO detection weights, auto-downloaded
img = read_image("path/to/image.jpg")
out = model(img)[0]
print(out["boxes"], out["scores"], [model.names[int(i)] for i in out["labels"]])
drawn = Visualizer(model)(img, out)           # annotated BGR copy, any task
```

`load_model` returns the same inference wrapper you would build by hand
(`TorchModel`, `TRTModel`, …), so its output contract is the wrapper's own.
"""

from dfine_seg.api.loader import SIZES, TASKS, load_model, pretrained_path, read_image
from dfine_seg.viz import Visualizer

__all__ = [
    "load_model",
    "read_image",
    "pretrained_path",
    "Visualizer",
    "SIZES",
    "TASKS",
    "__version__",
]

try:  # installed distribution
    from importlib.metadata import version

    __version__ = version("dfine-seg")
except Exception:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"
