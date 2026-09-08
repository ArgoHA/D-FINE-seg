# D-FINE-seg end2end GPU inference with C++

Pipeline: NVDEC -> TensorRT -> GPU annotate (boxes/masks) -> NVENC (annotated output video). Supported tasks: `detect` | `segment` | `sem_seg`.

Tested on RTX 5070ti on cityscapes and default values:

detect   | 812 fps
segment  | 756 fps
sem_seg  | 794 fps

## Usage

Default values:
``` bash
make cpp_e2e
```

Passing arguments:
``` bash
make cpp_e2e ARGS="--engine path/to/model.engine --videos path/to/videos"
```

Task type will be chosen automatically based on the passed .engine file. `--n-classes` only sizes the colour palette (engines carry no class count); the overlay draws no class names.

The engine must be exported with `train.keep_ratio: False` - the preprocess kernel squishes the frame to the engine input and has no letterbox path, same restriction as `TRTModel.gpu_run`.

`--help` lists the rest, including `--stage` for locating the bottleneck.
