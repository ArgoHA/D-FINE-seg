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
make cpp_e2e ARGS="--engine path/to/model.engine --videos path/to/videos --classes person,rider,car"
```

Task type will be chosen automatically based on the passed .engine file.
