.PHONY: main train split export bench infer test test-fast test_batching check_errors ov_int8 trt_int8

PY := uv run python

main:
	@$(MAKE) train
	$(PY) -m src.dl.export
	$(PY) -m src.dl.bench

split:
	$(PY) -m src.etl.split

train:
	@DDP_ENABLED=$$($(PY) -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg.get('train', {}).get('ddp', {}).get('enabled', False))" 2>/dev/null || echo "False"); \
	if [ "$$DDP_ENABLED" = "True" ] || [ "$$DDP_ENABLED" = "true" ]; then \
		NUM_GPUS=$$($(PY) -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); print(cfg.get('train', {}).get('ddp', {}).get('n_gpus', 2))" 2>/dev/null || echo "2"); \
		echo "🚀 Training with DDP using $$NUM_GPUS GPUs..."; \
		uv run torchrun --nproc_per_node=$$NUM_GPUS --master_port=29500 -m src.dl.train; \
	else \
		echo "🔧 Training with single GPU..."; \
		$(PY) -m src.dl.train; \
	fi

export:
	$(PY) -m src.dl.export

bench:
	$(PY) -m src.dl.bench

infer:
	$(PY) -m src.dl.infer

test_batching:
	$(PY) -m src.dl.test_batching

check_errors:
	$(PY) -m src.dl.check_errors

ov_int8:
	$(PY) -m src.dl.ov_int8

trt_int8:
	$(PY) -m src.dl.trt_int8

test:
	uv run pytest -q

test-fast:
	uv run pytest -q -m "not slow and not gpu"
