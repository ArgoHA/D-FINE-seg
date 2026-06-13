"""Pre-step diagnostic for ideas.md Tier-1 #5 (IoU-aware cls target).

Loads a trained checkpoint, runs the val set through the model + the SAME Hungarian
matcher used in training, and histograms the matched-pair IoU (diag IoU of matched
pred/gt) — exactly the `ious` that loss_labels_vfl builds its target from.

Heavy mass near IoU=0 among matches -> pick GCL (target stays informative at IoU=0);
otherwise IA-BCE. Run: uv run python -m experiments.diag_matched_iou
"""

import hydra
import torch
from omegaconf import DictConfig

from src.d_fine.arch.utils import box_cxcywh_to_xyxy, box_iou
from src.d_fine.dfine import build_loss, build_model
from src.dl.dataset import Loader

CKPT = "experiments/runs/baseline_h30/seed42/model.pt"
N_BATCHES = 60  # ~enough matched pairs for a stable histogram


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(cfg.train.label_to_name)

    model = build_model(
        cfg.model_name, num_classes, False, device,
        img_size=cfg.train.img_size, in_channels=cfg.train.in_channels,
    )
    model.load_state_dict(torch.load(CKPT, weights_only=True), strict=False)
    model.eval()
    matcher = build_loss(cfg.model_name, num_classes, 0.0, False).matcher

    loader = Loader(
        root_path=__import__("pathlib").Path(cfg.train.data_path),
        img_size=tuple(cfg.train.img_size), batch_size=8,
        num_workers=cfg.train.num_workers, cfg=cfg, debug_img_processing=False,
    )
    _, val_loader, _ = loader.build_dataloaders(distributed=False)

    all_ious = []
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(val_loader):
            if i >= N_BATCHES:
                break
            inputs = inputs.to(device)
            targets = [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()}
                       for t in targets]
            out = model(inputs)
            main = {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}
            indices = matcher(main, targets)["indices"]
            for b, (src, tgt) in enumerate(indices):
                if len(src) == 0:
                    continue
                iou, _ = box_iou(box_cxcywh_to_xyxy(out["pred_boxes"][b][src]),
                                 box_cxcywh_to_xyxy(targets[b]["boxes"][tgt]))
                all_ious.append(torch.diag(iou).cpu())

    ious = torch.cat(all_ious)
    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 1.01]
    print(f"\n=== matched-IoU histogram ({CKPT}) ===")
    print(f"matched pairs: {len(ious)}  mean {ious.mean():.3f}  median {ious.median():.3f}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ((ious >= lo) & (ious < hi)).float().mean().item()
        print(f"  [{lo:.1f}, {hi:.1f}): {m*100:5.1f}%  {'#'*round(m*50)}")
    print(f"\nIoU<0.1 mass (the GCL trigger): {(ious < 0.1).float().mean().item()*100:.1f}%")
    print(f"IoU<0.3 mass: {(ious < 0.3).float().mean().item()*100:.1f}%")


if __name__ == "__main__":
    main()
