import numpy as np

from dfine_seg.dl import check_errors


def test_wrong_class_match_is_saved_as_fp_and_fn(monkeypatch, tmp_path):
    saved = []

    def record(case_type, *args):
        saved.append((case_type, args[-1]))

    monkeypatch.setattr(check_errors, "save_case", record)
    check_errors.check_results(
        img=np.zeros((100, 100, 3), dtype=np.uint8),
        img_path=tmp_path / "sample.jpg",
        preds={
            "boxes": [[40, 40, 60, 60]],
            "labels": [1],
            "scores": [0.9],
        },
        targets={"boxes": [[0.5, 0.5, 0.2, 0.2]], "labels": [0]},
        iou_thresh=0.5,
        conf_thresh=0.5,
        output_dir=tmp_path,
        label_to_name={0: "cat", 1: "dog"},
    )
    assert saved == [("FPs", "pred_0"), ("FNs", "gt_0")]
