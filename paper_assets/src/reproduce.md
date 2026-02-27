# How to reproduce paper experiments

1. Download [TACO dataset](https://datasetninja.com/taco#download)
2. python paper_assets/taco_to_yolo.py --input_dir ... --output_dir ...
3. python paper_assets/split_by_batch.py --images_dir ... --labels_dir ... --output_dir ... --val_ratio 0.15
4. python -m src.etl.split_from_yolo (update the path to output dir from above)
5. cp all images and labels from yolo format to d-fine-seg dataset folder
6. python src/etl/polygot_to_bbox.py input_folder (with segmentation labels)

You should end up with 2 datasets with same images and labels, just different structure (one for D-FINE-seg, second one for YOLO):

```
data/
├── dataset/
│   ├── images/
│   ├── labels/
│   ├── train.csv
│   └── val.csv
└── yolo_data/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml
```

Train and export YOLO with python paper_asets/train_yolo.py
Train D-FINE-seg with `make train`, export with `make export`

Add YOLO with the model path to `make bench` and run it to get main metrics:
`python -m src.dl.bench model_name=n exp_name=seg_n`

For AP mterics run compute_coco_ap:
`python -m paper_assets.compute_coco_ap model_name=n exp_name=seg_n task=segment`
