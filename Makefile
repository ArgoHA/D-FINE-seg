.PHONY: main train split export bench infer demo test test-fast test_batching check_errors ov_int8 trt_int8 build

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
