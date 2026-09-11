import cv2
import numpy as np
import pytest

from dfine_seg.dl.infer import IMAGE_EXTS, _files, crops, figure_input_type


def test_input_files_are_filtered_and_sorted(tmp_path):
    for name in ("b.png", "a.jpg", ".hidden.png", "notes.txt"):
        (tmp_path / name).touch()
    assert [path.name for path in _files(tmp_path, IMAGE_EXTS)] == ["a.jpg", "b.png"]
    assert figure_input_type(tmp_path) == "image"


def test_mixed_input_types_are_rejected(tmp_path):
    (tmp_path / "image.jpg").touch()
    (tmp_path / "video.mp4").touch()
    with pytest.raises(ValueError, match="mixed"):
        figure_input_type(tmp_path)


def test_relative_crop_padding_is_not_mutated(tmp_path):
    paddings = {"w": 0.1, "h": 0.1}
    result = {"boxes": np.array([[2, 2, 6, 6]])}
    crops(np.zeros((10, 10, 3), dtype=np.uint8), result, paddings, tmp_path, "small")
    crops(np.zeros((20, 20, 3), dtype=np.uint8), result, paddings, tmp_path, "large")

    assert paddings == {"w": 0.1, "h": 0.1}
    assert cv2.imread(str(tmp_path / "crops" / "small_0.jpg")).shape[:2] == (6, 6)
    assert cv2.imread(str(tmp_path / "crops" / "large_0.jpg")).shape[:2] == (8, 8)
