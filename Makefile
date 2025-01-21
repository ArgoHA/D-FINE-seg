.PHONY: train

train:
	python -m src.dl.train
	python -m src.dl.export
	python -m src.dl.bench
