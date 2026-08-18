"""`load_model()` resolves weights and dispatches to a backend wrapper; it does not wrap them.

No model is built here — dispatch and resolution are tested through the seams
(`pretrained_path`, the backend table, `read_image`).
"""

from pathlib import Path

import numpy as np
import pytest

from dfine_seg import SIZES, TASKS, load_model, pretrained_path, read_image
from dfine_seg.data.coco_names import COCO_NAMES
from dfine_seg.loader import _BACKENDS


@pytest.fixture
def weights(tmp_path):
    """Dummy files so ensure_pretrained short-circuits instead of hitting the network."""
    for size in SIZES:
        (tmp_path / f"dfine_{size}_coco.pt").touch()
        (tmp_path / f"dfine_{size}_obj2coco.pt").touch()
        (tmp_path / f"dfine_seg_{size}_coco.pt").touch()
    return tmp_path


def test_coco_names_complete():
    assert len(COCO_NAMES) == 80
    assert list(COCO_NAMES) == list(range(80))
    assert COCO_NAMES[0] == "person"


def test_exports():
    assert SIZES == ("n", "s", "m", "l", "x")
    assert TASKS == ("detect", "segment", "sem_seg")


# ---- weight resolution -------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_detect_filenames(weights, size):
    assert pretrained_path(size, "detect", "coco", weights).endswith(f"dfine_{size}_coco.pt")


@pytest.mark.parametrize("size", SIZES)
def test_segment_filenames(weights, size):
    assert pretrained_path(size, "segment", "coco", weights).endswith(f"dfine_seg_{size}_coco.pt")


def test_obj2coco(weights):
    assert pretrained_path("s", "detect", "obj2coco", weights).endswith("dfine_s_obj2coco.pt")


def test_sem_seg_has_no_pretrained_weights(weights):
    with pytest.raises(ValueError, match="no pretrained sem_seg weights"):
        pretrained_path("s", "sem_seg", "coco", weights)


def test_unknown_size(weights):
    with pytest.raises(ValueError, match="size must be one of"):
        pretrained_path("xl", "detect", "coco", weights)


def test_unknown_task(weights):
    with pytest.raises(ValueError, match="task must be one of"):
        pretrained_path("s", "classify", "coco", weights)


def test_unknown_dataset(weights):
    with pytest.raises(ValueError, match="dataset must be coco\\|obj2coco"):
        pretrained_path("s", "detect", "imagenet", weights)


# ---- dispatch ----------------------------------------------------------------


def test_backend_table_covers_every_exported_format():
    assert set(_BACKENDS) == {".pt", ".engine", ".onnx", ".xml", ".mlpackage", ".tflite"}


def test_backend_classes_exist_and_are_named_correctly():
    """Guards against a typo'd class name that would only fail at load time."""
    from importlib import import_module

    for suffix, (module_name, class_name) in _BACKENDS.items():
        try:
            module = import_module(module_name)
        except ImportError:
            continue  # backend extra not installed in this env
        assert hasattr(module, class_name), f"{suffix}: {module_name} has no {class_name}"


def test_missing_file_message():
    with pytest.raises(FileNotFoundError, match="Pass a model path"):
        load_model("nope/missing.pt")


def test_unsupported_extension(tmp_path):
    bad = tmp_path / "model.bin"
    bad.touch()
    with pytest.raises(ValueError, match="unsupported model file"):
        load_model(bad)


def test_dispatch_passes_kwargs_and_path(tmp_path, monkeypatch):
    """load_model() constructs the backend with the path positionally and kwargs verbatim."""
    seen = {}

    class Fake:
        def __init__(self, path, **kw):
            seen["path"], seen["kw"] = path, kw

    import dfine_seg.loader as loader

    monkeypatch.setitem(loader._BACKENDS, ".pt", ("dfine_seg.loader", "Fake"))
    monkeypatch.setattr(loader, "Fake", Fake, raising=False)
    ckpt = tmp_path / "model.pt"
    ckpt.touch()

    load_model(ckpt, conf_thresh=0.3, input_height=960)
    assert seen["path"] == str(ckpt)
    assert seen["kw"] == {"conf_thresh": 0.3, "input_height": 960}


def test_task_forwarded_for_path_loads(tmp_path, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, path, **kw):
            seen.update(kw)

    import dfine_seg.loader as loader

    monkeypatch.setitem(loader._BACKENDS, ".pt", ("dfine_seg.loader", "Fake"))
    monkeypatch.setattr(loader, "Fake", Fake, raising=False)
    ckpt = tmp_path / "model.pt"
    ckpt.touch()

    load_model(ckpt, task="segment")
    assert seen["task"] == "segment"


@pytest.mark.parametrize("suffix", [".onnx", ".engine", ".xml", ".mlpackage", ".tflite"])
def test_task_not_forwarded_to_graph_backends(tmp_path, monkeypatch, suffix):
    """Only TorchModel takes task=; every graph wrapper would raise TypeError on it."""
    seen = {}

    class Fake:
        def __init__(self, path, **kw):
            seen.update(kw)

    import dfine_seg.loader as loader

    monkeypatch.setitem(loader._BACKENDS, suffix, ("dfine_seg.loader", "Fake"))
    monkeypatch.setattr(loader, "Fake", Fake, raising=False)
    artifact = tmp_path / f"model{suffix}"
    artifact.touch()

    load_model(artifact, task="segment")
    assert "task" not in seen


def _wrapper_for(suffix):
    import inspect as _inspect
    from importlib import import_module

    from dfine_seg.loader import _BACKENDS

    module_name, class_name = _BACKENDS[suffix]
    try:
        module = import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} not installed")
    return class_name, _inspect.signature(getattr(module, class_name)).parameters


@pytest.mark.parametrize("suffix", [".onnx", ".engine", ".mlpackage", ".xml"])
def test_fused_backends_have_no_class_count(suffix):
    """The graph emits labels and a per-class list carries its own length: nothing to pass."""
    class_name, params = _wrapper_for(suffix)
    assert "n_outputs" not in params, f"{class_name} still takes n_outputs"


def test_litert_keeps_an_optional_class_count():
    """LiteRT decodes labels with it (`topk_idx % n_outputs`) — real, but read from the graph."""
    class_name, params = _wrapper_for(".tflite")
    assert params["n_outputs"].default is None, f"{class_name}.n_outputs is required"


@pytest.mark.parametrize("suffix", [".onnx", ".engine", ".mlpackage", ".xml", ".tflite"])
def test_second_positional_is_conf_thresh(suffix):
    """Guards the removal: a stale `Model(path, 80)` must not silently become a threshold."""
    _, params = _wrapper_for(suffix)
    second = list(params)[1]  # signature() on a class drops self, so [0] is model_path
    expected = "conf_thresh" if suffix != ".tflite" else "n_outputs"
    assert second == expected


# ---- names -------------------------------------------------------------------


def _fake_backend(monkeypatch, wrapper_names=None):
    class Fake:
        def __init__(self, path, **kw):
            if wrapper_names is not None:
                self.names = wrapper_names

    import dfine_seg.loader as loader

    monkeypatch.setitem(loader._BACKENDS, ".pt", ("dfine_seg.loader", "Fake"))
    monkeypatch.setattr(loader, "Fake", Fake, raising=False)


def test_released_filename_gets_coco_names(tmp_path, monkeypatch):
    """load_model("pretrained/dfine_n_coco.pt") must match load_model("n")."""
    _fake_backend(monkeypatch)
    ckpt = tmp_path / "dfine_n_coco.pt"
    ckpt.touch()
    assert load_model(ckpt).names == COCO_NAMES


def test_unknown_filename_gets_no_names(tmp_path, monkeypatch):
    _fake_backend(monkeypatch)
    ckpt = tmp_path / "my_model.pt"
    ckpt.touch()
    assert load_model(ckpt).names is None


def test_wrapper_names_beat_the_coco_default(tmp_path, monkeypatch):
    """Names read from a sidecar config outrank the bundled COCO map."""
    _fake_backend(monkeypatch, wrapper_names={0: "cat"})
    ckpt = tmp_path / "dfine_n_coco.pt"
    ckpt.touch()
    assert load_model(ckpt).names == {0: "cat"}


def test_explicit_names_beat_everything(tmp_path, monkeypatch):
    _fake_backend(monkeypatch, wrapper_names={0: "cat"})
    ckpt = tmp_path / "dfine_n_coco.pt"
    ckpt.touch()
    assert load_model(ckpt, names={0: "dog"}).names == {0: "dog"}


# ---- read_image --------------------------------------------------------------


def test_read_image_npy_roundtrip(tmp_path):
    arr = np.random.randint(0, 255, (8, 6, 4), dtype=np.uint8)
    p = tmp_path / "x.npy"
    np.save(p, arr)
    assert np.array_equal(read_image(p), arr)  # byte-faithful, incl. 4-channel stacks


def test_read_image_jpg_is_bgr(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # pure blue in BGR
    p = tmp_path / "b.png"
    cv2.imwrite(str(p), img)
    out = read_image(p)
    assert out.shape == (4, 4, 3)
    assert out[0, 0, 0] == 255 and out[0, 0, 2] == 0


def test_read_image_pil_is_rgb():
    Image = pytest.importorskip("PIL.Image")
    pil = Image.new("RGB", (4, 4), (255, 0, 0))  # pure red
    out = read_image(pil)
    assert out[0, 0, 0] == 255 and out[0, 0, 2] == 0


def test_read_image_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="could not read image"):
        read_image(tmp_path / "nope.jpg")


def test_read_image_accepts_str_and_path(tmp_path):
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    p = tmp_path / "x.npy"
    np.save(p, arr)
    assert read_image(str(p)).shape == read_image(Path(p)).shape
