# D-FINE-seg C++ inference

Two targets from one source tree:

| | what it is | deps beyond CUDA/TensorRT | build |
|---|---|---|---|
| **core** (`libdfine.so` + `include/dfine.h`) | engine + fused pre/postprocess kernels. Embed it in your software. | none | `make cpp` |
| **video** (`dfine_e2e`) | NVDEC -> core -> GPU overlay -> NVENC, annotated mp4 out. | ffmpeg, NVDEC, NVENC | `make cpp_e2e` |

The core is what ships to a robot or a drone: `readelf -d libdfine.so` lists only `libnvinfer.so.10`, `libcudart.so.12` and libc. The video app is one *consumer* of it.

Both tasks follow the engine: `labels/boxes/scores` = detect, `+ masks` = instance segmentation, a single int32 map = semantic segmentation.

## core

```c
#include <dfine.h>

DfineModel* m = dfine_open("model.engine", /*device=*/0);
DfineSlot*  s = dfine_slot_create(m);

DfineDets d;
dfine_infer(m, s, img, w, h, /*pitch=*/0, DFINE_BGR8, /*on_device=*/0, &d);
for (int i = 0; i < d.count; ++i)
    use(d.labels[i], d.scores[i], &d.boxes[i * 4]);   // boxes in your image's pixels
```

That is the whole blocking path. For throughput, submit without syncing and let the GPU stay fed while your code handles the previous frame:

```c
dfine_infer_async(m, slot[i], img, w, h, 0, DFINE_BGR8, 1, stream[i]);
// ... results are DEVICE pointers: dfine_dets / dfine_masks / dfine_sem,
//     valid once dfine_event(slot[i]) has fired, until the next infer into that slot.
```

A **slot** is one execution context - its own TRT context, CUDA graph and buffers. One slot per thread for concurrency; two or three per thread to pipeline. Create them before the threads that use them start issuing other CUDA work (slot creation captures a CUDA graph).

`examples/bench.cpp` measures this path with no video pipeline in the way - the number to size an
embedded target with:

```bash
./cpp/build/dfine_bench model.engine                        # 1920x1080, 400 iters, 1/2/4/8 slots
./cpp/build/dfine_bench model.engine --width 1280 --height 720 --slots 1,4
```

RTX 5070 Ti, 1920x1080 frame already on the GPU:

| | detect | segment | sem_seg |
|---|---|---|---|
| blocking, 1 slot | 1.100 ms (909 fps) | 1.580 ms (633 fps) | 1.281 ms (780 fps) |
| async, 4 slots | 0.657 ms (1522 fps) | 1.153 ms (867 fps) | 1.059 ms (944 fps) |

**Nothing is copied to the host for you, and nothing is upsampled for you.** Detections are ~8 KB, so `dfine_copy_dets` is cheap; masks come back on the engine's own grid and `dfine_upsample_masks` is opt-in, because a full-res upsample the caller did not want is waste (300 instances at 1080p is 1.2 GB; the engine grid is 15 MB). Same reasoning as the Python wrappers, which never force `.cpu()` either.
Boxes come out in the pixel space of the image you passed. Engines must be exported with `train.keep_ratio: False` - the preprocess kernel squishes and has no letterbox path, the same restriction as `TRTModel.gpu_run`.

`examples/detect.cpp` is a complete consumer (`dfine_example model.engine frame.ppm`); both examples build with the core, no ffmpeg needed.

## video

```bash
make cpp_e2e                                             # defaults
make cpp_e2e ARGS="--engine path/model.engine --videos path/to/videos"
```

RTX 5070 Ti, Cityscapes clips: detect 812 fps, segment 756 fps, sem_seg 794 fps.
`--n-classes` only sizes the colour palette (engines carry no class count); the overlay draws no class names. `--stage parse|decode|infer|draw|full|bench` runs a prefix of the pipeline, for locating the bottleneck. `--help` lists the rest.

## building for another GPU

CUDA architecture is baked in at compile time and a build does **not** run on an older arch (the embedded PTX only JITs forward). Set the deployment GPU:

```bash
make cpp CUDA_ARCH=87     # Jetson Orin;  89 = Ada,  86 = Ampere,  120 = RTX 50xx (default)
```

Everything else is ordinary ABI matching: the engine itself is GPU- and TensorRT-version specific (build it on the target), and the `.so` must find the same TensorRT major version it was built against.

## parity

Core results are identical with the Python wrappers on the same frame, verified per task: detect and segment boxes to 0.000 px, `dfine_upsample_masks` 0 differing pixels out of 39.8 M vs `TRTModel.process_masks` + `cleanup_masks`, sem_seg label map 100 % pixel agreement vs `TRTModel.gpu_run`. The video path is byte-identical to its pre-split output on all three tasks (`--dump` / `--dump-map`).
