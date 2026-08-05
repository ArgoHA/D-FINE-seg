"""Pin `make split` on a single-file COCO json.

Mirrors the YOLO path: image-level split by the same ratios/seed, written as standalone
COCO files. Getting this wrong leaks images across splits or orphans annotations.
"""

import json

import pytest

from src.dl.dataset import load_coco_split
from src.etl.split import split, split_coco


def _coco(n_images=20, empty_ids=()):
    """n_images, each with 2 anns unless its id is in empty_ids."""
    images = [
        {"id": i, "file_name": f"img_{i:03d}.jpg", "width": 64, "height": 48}
        for i in range(n_images)
    ]
    anns, aid = [], 0
    for img in images:
        if img["id"] in empty_ids:
            continue
        for _ in range(2):
            aid += 1
            anns.append(
                {
                    "id": aid,
                    "image_id": img["id"],
                    "category_id": 1,
                    "bbox": [1, 2, 10, 10],
                    "segmentation": [[1, 2, 11, 2, 11, 12, 1, 12]],
                }
            )
    return {
        "info": {"description": "fixture"},
        "licenses": [{"id": 1, "name": "cc"}],
        "categories": [{"id": 1, "name": "person"}],
        "images": images,
        "annotations": anns,
    }


def _write(tmp_path, coco):
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    return tmp_path / "coco.json"


def _read(tmp_path, name):
    return json.loads((tmp_path / f"{name}.json").read_text())


def test_train_val_only_when_no_test_ratio(tmp_path):
    src = _write(tmp_path, _coco(20))
    split_coco(tmp_path, src, 0.85, 0.15, False, seed=42, shuffle=True)

    assert not (tmp_path / "test.json").exists()
    train, val = _read(tmp_path, "train"), _read(tmp_path, "val")
    # 16/4, not 17/3: 1 - 0.85 == 0.15000000000000002, so sklearn rounds 20*that up to 4.
    # Shared with the YOLO path (both call split_indices), which has always behaved this way.
    assert len(train["images"]) == 16 and len(val["images"]) == 4


def test_test_split_written_when_ratios_leave_room(tmp_path):
    src = _write(tmp_path, _coco(20))
    split_coco(tmp_path, src, 0.5, 0.25, False, seed=42, shuffle=True)

    counts = {n: len(_read(tmp_path, n)["images"]) for n in ("train", "val", "test")}
    assert sum(counts.values()) == 20
    assert counts["train"] == 10


def test_images_are_disjoint_and_complete(tmp_path):
    src = _write(tmp_path, _coco(20))
    split_coco(tmp_path, src, 0.5, 0.25, False, seed=42, shuffle=True)

    ids = [{img["id"] for img in _read(tmp_path, n)["images"]} for n in ("train", "val", "test")]
    assert set.intersection(*ids) == set()
    assert set.union(*ids) == set(range(20))
    assert sum(len(s) for s in ids) == 20  # no image in two splits


def test_annotations_follow_their_images(tmp_path):
    src = _write(tmp_path, _coco(20))
    split_coco(tmp_path, src, 0.5, 0.25, False, seed=42, shuffle=True)

    seen = set()
    for name in ("train", "val", "test"):
        d = _read(tmp_path, name)
        img_ids = {img["id"] for img in d["images"]}
        assert {a["image_id"] for a in d["annotations"]} <= img_ids  # no orphans
        assert len(d["annotations"]) == 2 * len(d["images"])
        ann_ids = {a["id"] for a in d["annotations"]}
        assert not (ann_ids & seen)  # no annotation duplicated across splits
        seen |= ann_ids
    assert len(seen) == 40


def test_shared_keys_preserved(tmp_path):
    src = _write(tmp_path, _coco(20))
    split_coco(tmp_path, src, 0.85, 0.15, False, seed=42, shuffle=True)

    for name in ("train", "val"):
        d = _read(tmp_path, name)
        assert d["categories"] == [{"id": 1, "name": "person"}]
        assert d["info"] == {"description": "fixture"}
        assert d["licenses"] == [{"id": 1, "name": "cc"}]


def test_ignore_negatives_drops_annotationless_images(tmp_path):
    src = _write(tmp_path, _coco(20, empty_ids=(0, 1, 2, 3)))
    split_coco(tmp_path, src, 0.85, 0.15, True, seed=42, shuffle=True)

    kept = {img["id"] for n in ("train", "val") for img in _read(tmp_path, n)["images"]}
    assert kept == set(range(4, 20))


def test_negatives_kept_by_default(tmp_path):
    src = _write(tmp_path, _coco(20, empty_ids=(0, 1, 2, 3)))
    split_coco(tmp_path, src, 0.85, 0.15, False, seed=42, shuffle=True)

    kept = {img["id"] for n in ("train", "val") for img in _read(tmp_path, n)["images"]}
    assert kept == set(range(20))


def test_same_seed_is_deterministic(tmp_path):
    src = _write(tmp_path, _coco(20))
    split_coco(tmp_path, src, 0.85, 0.15, False, seed=42, shuffle=True)
    first = (tmp_path / "train.json").read_text()
    split_coco(tmp_path, src, 0.85, 0.15, False, seed=42, shuffle=True)
    assert (tmp_path / "train.json").read_text() == first


def test_no_shuffle_is_sorted_by_file_name(tmp_path):
    coco = _coco(20)
    coco["images"].reverse()  # source order must not matter
    src = _write(tmp_path, coco)
    split_coco(tmp_path, src, 0.85, 0.15, False, seed=42, shuffle=False)

    train = _read(tmp_path, "train")
    assert [img["file_name"] for img in train["images"]] == sorted(
        img["file_name"] for img in train["images"]
    )


def test_output_is_loadable_by_the_dataset(tmp_path):
    """The whole point: the loader must consume what split wrote."""
    src = _write(tmp_path, _coco(20))
    split_coco(tmp_path, src, 0.85, 0.15, False, seed=42, shuffle=True)

    entries, cat_map = load_coco_split(tmp_path / "train.json")
    assert len(entries) == 16
    assert cat_map == {1: 0}
    assert all(e["targets"].shape[0] == 2 for e in entries)
    assert all(len(e["polys_abs"]) == 2 for e in entries)


def test_matches_the_yolo_path_image_for_image(tmp_path):
    """The requirement: same split as YOLO, just from a coco.json.

    Compared with shuffle=False, where both paths order deterministically (YOLO sorts the
    filenames it globs, COCO sorts by file_name); with shuffle=True the YOLO path inherits
    an arbitrary `iterdir()` order, so only its ratios are comparable, not its membership.
    """
    coco = _coco(20)
    names = [img["file_name"] for img in coco["images"]]

    yolo = tmp_path / "yolo"
    (yolo / "images").mkdir(parents=True)
    (yolo / "labels").mkdir()
    for n in names:
        (yolo / "images" / n).touch()
        (yolo / "labels" / f"{n[:-4]}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    split(yolo, 0.5, 0.25, yolo / "images", False, seed=42, shuffle=False)

    cc = tmp_path / "coco"
    cc.mkdir()
    split_coco(cc, _write(cc, coco), 0.5, 0.25, False, seed=42, shuffle=False)

    for name in ("train", "val", "test"):
        from_yolo = (yolo / f"{name}.csv").read_text().split()
        from_coco = [img["file_name"] for img in _read(cc, name)["images"]]
        assert from_yolo == from_coco, name


def test_missing_coco_json_raises(tmp_path):
    # main() raises its own FileNotFoundError with a fix-it message before reaching here
    with pytest.raises(FileNotFoundError):
        split_coco(tmp_path, tmp_path / "coco.json", 0.85, 0.15, False, seed=42, shuffle=True)
