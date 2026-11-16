"""`dfine_seg.export()`: config-free export from a checkpoint, the path `dfine export` maps onto."""

from pathlib import Path

import pytest
import torch

from tests.unit.test_ckpt_inspect import fake_sd

pytest.importorskip("pandas")  # dl/export pulls the training stack


def test_top_level_name_is_the_export_function():
    import dfine_seg
    from dfine_seg.dl.export import export

    assert dfine_seg.export is export


def test_unknown_format_is_refused(tmp_path):
    from dfine_seg import export

    with pytest.raises(ValueError, match="unknown export format"):
        export(tmp_path / "missing.pt", ["onnx", "tflite"])


def test_task_mismatch_is_refused(tmp_path):
    from dfine_seg import export

    p = tmp_path / "sem.pt"
    torch.save(fake_sd(task="sem_seg", num_classes=19), p)
    with pytest.raises(ValueError, match="sem_seg checkpoint, not detect"):
        export(p, task="detect")


def _n_spec(sd, **overrides):
    from dfine_seg.api.ckpt import describe

    return {**describe(sd), "img_size": (320, 320), **overrides}


def test_detect_from_segment_weights_leaves_the_mask_head_behind():
    from dfine_seg.dl.export import prepare_model
    from dfine_seg.model.dfine import build_model

    sd = build_model("n", 3, enable_mask_head=True, device="cpu", task="segment").state_dict()
    model = prepare_model(_n_spec(sd, task="detect"), sd, "cpu")
    assert not any(k.startswith("decoder.mask") for k in model.state_dict())


def test_weights_missing_from_the_checkpoint_are_refused():
    from dfine_seg.dl.export import prepare_model
    from dfine_seg.model.dfine import build_model

    sd = build_model("n", 3, enable_mask_head=False, device="cpu").state_dict()
    spec = _n_spec(sd)
    del sd["decoder.enc_bbox_head.layers.0.weight"]
    with pytest.raises(ValueError, match="does not match"):
        prepare_model(spec, sd, "cpu")


def test_img_size_precedence(tmp_path):
    """explicit argument -> checkpoint meta -> sidecar config.yaml -> 640."""
    from dfine_seg.dl.export import _resolve
    from dfine_seg.model.utils import save_checkpoint

    p = tmp_path / "model.pt"
    torch.save(fake_sd(), p)
    assert _resolve(p, None, None)[1]["img_size"] == (640, 640)
    (tmp_path / "config.yaml").write_text("train:\n  img_size: [480, 800]\n")
    assert _resolve(p, None, None)[1]["img_size"] == (480, 800)
    save_checkpoint(p, fake_sd(), {"img_size": [320, 384]})
    assert _resolve(p, None, None)[1]["img_size"] == (320, 384)
    assert _resolve(p, None, (256, 512))[1]["img_size"] == (256, 512)
    assert _resolve(p, None, 416)[1]["img_size"] == (416, 416)


def _config(tmp_path, *overrides):
    from hydra import compose, initialize_config_dir

    from dfine_seg.config.resolve import DEFAULT_CONFIG

    run = tmp_path / "run"
    run.mkdir()
    with initialize_config_dir(config_dir=str(DEFAULT_CONFIG.parent), version_base=None):
        cfg = compose(
            config_name=DEFAULT_CONFIG.stem,
            overrides=[f"train.root={tmp_path}", f"train.path_to_save={run}", *overrides],
        )
    return cfg, run


def test_config_maps_onto_export(tmp_path, monkeypatch):
    """`dfine export` is export() with the config's values, writing run/model.*."""
    from dfine_seg.dl import export as mod

    cfg, run = _config(tmp_path, "export.formats=null", "export.max_batch_size=4")
    torch.save(fake_sd(num_classes=len(cfg.train.label_to_name)), run / "model.pt")
    calls = {}
    monkeypatch.setattr(mod, "export", lambda ckpt, formats, **kw: calls.update(kw, ckpt=ckpt))
    mod.run_from_config(cfg)

    assert calls["ckpt"] == run / "model.pt"
    assert (Path(calls["output_dir"]), calls["name"]) == (run, "model")
    assert calls["max_batch_size"] == 4 and calls["half"] is True and calls["parity"] is True
    assert list(calls["img_size"]) == list(cfg.train.img_size) and calls["task"] == "detect"


def test_trained_run_must_match_its_config(tmp_path):
    """Export builds what the weights are; a config edited after training would make bench and
    infer read the wrong class names, so the mismatch is refused."""
    from dfine_seg.dl.export import run_from_config

    cfg, run = _config(tmp_path)
    torch.save(fake_sd(num_classes=len(cfg.train.label_to_name) + 1), run / "model.pt")
    with pytest.raises(ValueError, match="Export with the config model.pt was trained with"):
        run_from_config(cfg)


@pytest.mark.gpu
def test_tensorrt_parse_failure_raises(tmp_path):
    pytest.importorskip("tensorrt")
    from dfine_seg.dl.export import export_to_tensorrt

    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx graph")
    with pytest.raises(RuntimeError, match="TensorRT could not parse"):
        export_to_tensorrt(bad, half=True, max_batch_size=1)


@pytest.mark.slow
def test_export_needs_only_the_checkpoint(tmp_path):
    """No config and no run dir: the input size comes from the checkpoint's own meta. OpenVINO's
    raw-head intermediate must not be left behind as <name>.onnx."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    pytest.importorskip("openvino")
    from dfine_seg import export
    from dfine_seg.model.dfine import build_model
    from dfine_seg.model.utils import save_checkpoint

    sd = build_model("n", 3, enable_mask_head=False, device="cpu").state_dict()
    save_checkpoint(tmp_path / "w.pt", sd, {"img_size": [320, 384]})
    out = tmp_path / "out"
    paths = export(tmp_path / "w.pt", ["openvino"], device="cpu", output_dir=out, name="w.v2")

    assert paths == {"openvino": out / "w.v2.xml"}
    assert sorted(p.name for p in out.iterdir()) == ["w.v2.bin", "w.v2.xml"]

    paths = export(tmp_path / "w.pt", "onnx", device="cpu")
    assert paths == {"onnx": tmp_path / "w.onnx"}
    sess = ort.InferenceSession(str(paths["onnx"]), providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].shape == [1, 3, 320, 384]
    assert [o.name for o in sess.get_outputs()] == ["labels", "boxes", "scores"]
