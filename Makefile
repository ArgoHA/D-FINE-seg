.PHONY: main train split export bench infer demo test test-fast test_batching check_errors ov_int8 trt_int8 build cpp cpp_e2e

# The `dfine` console script is installed by `uv sync` and handles DDP itself.
CLI := uv run dfine

main:
	@$(MAKE) train
	$(CLI) export
	$(CLI) bench

split:
	$(CLI) split

train:
	$(CLI) train

export:
	$(CLI) export

bench:
	$(CLI) bench

infer:
	$(CLI) infer

demo:
	$(CLI) demo

test_batching:
	$(CLI) test-batching

check_errors:
	$(CLI) check-errors

ov_int8:
	$(CLI) ov-int8

trt_int8:
	$(CLI) trt-int8

test:
	uv run pytest -q

test-fast:
	uv run pytest -q -m "not slow and not gpu"

build:
	rm -rf dist
	uv build

# Core library (libdfine.so + dfine.h) only - what you cross-build for a robot/drone target.
# Set the deployment GPU arch: make cpp CUDA_ARCH=87 (Jetson Orin), 89 (Ada), 86 (Ampere).
CUDA_ARCH ?=
cpp:
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DDFINE_BUILD_VIDEO=OFF \
	  $(if $(CUDA_ARCH),-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH),)
	cmake --build cpp/build -j

cpp_e2e:
	cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DDFINE_BUILD_VIDEO=ON \
	  $(if $(CUDA_ARCH),-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH),)
	cmake --build cpp/build -j
	cpp/build/dfine_e2e $(ARGS)
