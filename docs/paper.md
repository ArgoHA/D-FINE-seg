---
title: 'D-FINE-seg: a PyTorch framework for real-time object detection, instance segmentation, and semantic segmentation with multi-backend deployment'
tags:
  - Python
  - computer vision
  - object detection
  - instance segmentation
  - semantic segmentation
  - deep learning
  - model deployment
authors:
  - name: Argo Saakyan
    orcid: 0009-0009-7577-715X
    affiliation: 1
    corresponding: true
affiliations:
  - name: Agnify.ai, Spain
    index: 1
date: 18 October 2026
bibliography: paper.bib
---

# Summary

Object detection locates and classifies objects in an image; instance and semantic segmentation go further and outline each object or label every pixel - capabilities needed in fields from drone-based monitoring to document analysis. D-FINE-seg is a pip-installable Python framework that takes a computer vision project from raw annotations to a deployed inference engine. It trains and runs real-time models for three tasks - object detection, instance segmentation, and semantic segmentation - switched by a single configuration flag, in five model sizes with pretrained weights for each task (COCO [@lin2014coco] for detection and instance segmentation, Cityscapes [@cordts2016cityscapes] for semantic segmentation - those weights inherit Cityscapes' non-commercial terms). It exports to ONNX, TensorRT, OpenVINO, and CoreML, with an automated parity check of each exported backend against the PyTorch reference, and runs inference through one Python API on GPU and CPU.

# Statement of need

D-FINE-seg closes the gap between a strong detection architecture and an applied segmentation project. Upstream D-FINE [@peng2024dfine] ships detection only: open issues on its tracker request instance segmentation and segmentation-style polygon outputs ([#149](https://github.com/Peterande/D-FINE/issues/149), [#174](https://github.com/Peterande/D-FINE/issues/174)), and there are no segmentation heads or mask-aware losses. Building segmentation on this model family means writing heads, losses, and deployment glue from scratch.

Applied vision projects run a tight loop: annotate, train, inspect errors, fix annotations, retrain, export, deploy. D-FINE-seg ships this loop as one toolchain: training with debugging outputs (augmented batches, validation overlays of predictions against ground truth), a check-errors command that isolates model errors for annotation review, and export with parity checks. Four-channel input (RGB plus thermal [@liu2022m3fd], depth, or near-infrared) is a native code path for multi-modal sensors. The loop reflects the author's production workflow. The intended audience is applied computer-vision researchers who need real-time detection or segmentation in their own domain - drone imagery [@zhu2021visdrone], environmental monitoring [@proenca2020taco], multi-modal RGB+thermal sensing - and practitioners shipping vision models to server and edge hardware.

# State of the field

Real-time DETR-family detectors - RT-DETR [@zhao2024rtdetr] and D-FINE [@peng2024dfine], building on DETR [@carion2020detr] - compete with YOLO-style models, while strong segmentation baselines such as Mask2Former and SegFormer live in research codebases [@cheng2022mask2former; @xie2021segformer]. Among deployable alternatives: MMDetection [@chen2019mmdetection] and Detectron2 [@wu2019detectron2] are research-first toolboxes where deployment is secondary; Ultralytics [@jocher2026yolo26] ships YOLO26 with a semantic-segmentation variant and wide backend export, under an AGPL-3.0 licence; RF-DETR [@robinson2025rfdetr] offers a DETR-family detector with instance-segmentation and keypoint variants, but no semantic segmentation. D-FINE-seg differs in the combination of: one D-FINE (DETR-family) architecture across all three tasks; automated cross-backend parity checks; the error-review loop for annotation iteration; Apache-2.0 licensing; and native four-channel input with pretrained-weight adaptation.

D-FINE-seg is a standalone framework rather than a fork of the upstream repository: upstream fine-tuning means cloning the repo and hand-editing its configs, while here the workflow is `pip install dfine-seg` plus a `dfine` CLI (`init`, `split`, `train`, `export`, `bench`) with a fuller training and debugging stack, and segmentation on top. The detection core reuses the D-FINE authors' Apache-2.0 model code with small, weight-compatible modifications; the framework around it - data handling, training loop, export pipeline, inference backends, CLI - is new code. That scope exceeds what a reasonable upstream contribution could carry - it was a framework rewrite around the model, not a head to graft on - so it lives as a separate project instead.

# Software design

One codebase serves three tasks through a single task flag. The instance segmentation head follows the Mask DINO paradigm [@li2023maskdino]: per-query mask embeddings from a small MLP are dotted with shared multi-scale mask features. Training is mask-aware (box-cropped BCE and Dice losses, mask costs in the Hungarian matcher, mask supervision inside contrastive denoising [@zhang2023dino]); detection models carry no mask compute - the mask head runs only in the segmentation tasks. The semantic segmentation head reuses the same mask fuser on full-frame features, adds a small convolutional neck and a 1x1 classifier - no queries, no NMS - and exports as a fused-argmax graph with a single integer label-map output.

Training prioritizes practical iteration: Muon/Adan optimization [@jordan2024muon; @xie2024adan], multi-GPU mixed-precision training with automatic batch-size probing, knowledge distillation from a frozen teacher, mosaic and Albumentations augmentation [@buslaev2020albumentations], and Weights & Biases tracking.

Deployment is a first-class subsystem: five backends (PyTorch [@paszke2019pytorch], ONNX Runtime [@onnx], TensorRT [@tensorrt], OpenVINO [@openvino], CoreML [@coremltools]), with `load_model` recovering size, task, and class count from the checkpoint itself. Exports cover the four graph formats, with INT8 variants for TensorRT and OpenVINO (detection and instance segmentation) and CoreML; a parity self-check then compares each exported backend against the PyTorch reference on a shared input - top-K score agreement for detection and instance segmentation, per-pixel label agreement for semantic segmentation - and writes it next to the weights. A small C++/CUDA core (`libdfine.so`, TensorRT inference with fused pre- and postprocessing kernels, plus an NVDEC->NVENC video pipeline) ships the same three-task inference outside Python. Two trade-offs are deliberate: the graph-backend wrappers duplicate code so each file can be copied into an application on its own, and the semantic head applies argmax before a nearest-neighbour resize to the original image resolution - upsampling logits first improves mIoU only marginally while substantially increasing TensorRT latency.

Continuous integration runs linting, a fast CPU test suite, install checks, and a predict smoke test; fixture-based regression tests over model outputs run locally via `make test`; and for any change that can affect model accuracy, a scripted regression (`scripts/regression_test.py`) retrains each task on Cityscapes and compares training, export, and benchmark metrics against tracked reference runs with explicit tolerances. A simplified ByteTrack tracker [@zhang2022bytetrack], SAM 3 auto-labeling [@carion2025sam3], and a Gradio demo complete the toolchain.

# Research impact statement

D-FINE-seg is in active research use. The companion preprint [@saakyan2026dfineseg] reports detection and instance-segmentation experiments on VisDrone [@zhu2021visdrone] and TACO [@proenca2020taco] produced with the framework; a public 28-model Cityscapes benchmark [@cordts2016cityscapes] ([github.com/ArgoHA/cityscapes-benchmark](https://github.com/ArgoHA/cityscapes-benchmark)) evaluates it against official checkpoints of four other model families, with the largest model reaching 84.84 mIoU at 14.64 ms end-to-end on native-resolution val (83.23 mIoU on the official test server; RTX 5070 Ti, TensorRT FP16, batch 1); outside groups build on it - the University of Canterbury's UC Vision group forked it for vineyard scanning and yield estimation ([uc-vision/D-FINE-seg](https://github.com/uc-vision/D-FINE-seg)), the author of the kraken text-recognition engine adapted it for historical document layout analysis ([dfine_kraken](https://github.com/mittagessen/dfine_kraken), published on PyPI as the basis for kraken integration), and users train it for text-line segmentation of manuscripts ([#36](https://github.com/ArgoHA/D-FINE-seg/issues/36)); and 56 of its 57 GitHub issues were opened by outside users. It is also deployed commercially for document processing (Veryfi Inc.), video monitoring and industrial inspection (Agnify.ai). Adoption at the time of writing: roughly 1,500 PyPI downloads, roughly 1,400 all-time downloads of the pretrained weights on Hugging Face (828 in the last 30 days), 217 GitHub stars and 36 forks.

# AI usage disclosure

From its first commit in January 2025 through October 2025, D-FINE-seg was written by the author without AI agents; this period produced the detection framework. AI use in that period was limited to inline code completion (GitHub Copilot in VS Code) and occasional questions to ChatGPT (models not recorded). Since November 2025, AI coding agents have been used for parts of the implementation, refactoring, test scaffolding, debugging, and documentation: GitHub Copilot in VS Code (Claude Opus 4.5, 4.6), Claude Code (Claude Opus 4.7, 4.8, 5.0, 5.5; Fable 5.0, 5.1), OpenAI Codex (Astra, Sol 6.1), and Prime Agent (GLM 5.3 Flash). The core codebase, including the instance-segmentation head, was written directly by the author in VS Code with AI assistance in some places; the author designed the architectures, made all design decisions, and reviewed, tested, and validated all AI-assisted code before release.

This manuscript was drafted and revised with AI assistance (Prime Agent, GLM 5.3 Flash; Claude Code with Claude Opus 5.5, which reviewed drafts against the JOSS requirements and suggested revisions) and reviewed, edited, and validated by the author, who is responsible for its content. No generative AI was used in communications with JOSS editors or reviewers.

# Acknowledgements

The detection core reuses the D-FINE authors' Apache-2.0 implementation, and the mask head design follows Mask DINO - the author thanks both teams. This work received no external funding.

# Conflicts of interest

The author was employed by Veryfi Inc. from 2023 to 2026 and is currently a Principal Computer Vision Engineer at Agnify.ai; D-FINE-seg is deployed in production at both companies (see the Research impact statement). Apart from this employment, the author declares no competing financial, personal, or professional interests.

# References
