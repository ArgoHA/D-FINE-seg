"""The exported graph samples with gathers instead of GridSample (TensorRT mis-executes the
latter); both paths must compute the same thing."""

import pytest
import torch
import torch.nn.functional as F

from dfine_seg.model.arch.utils import bilinear_gather, deformable_attention_core_func_v2


def test_bilinear_gather_matches_grid_sample():
    torch.manual_seed(0)
    n, c, h, w = 3, 5, 7, 11  # non-square: catches swapped axes
    value = torch.randn(n, c, h, w)
    # beyond [0, 1] on purpose: out-of-image corners must contribute zeros
    locations = torch.rand(n, 13, 4, 2) * 1.4 - 0.2
    locations[0, 0] = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5 / w, 0.5 / h], [-1.0, 2.0]])

    want = F.grid_sample(
        value, 2 * locations - 1, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    got = bilinear_gather(value.flatten(2), locations, h, w)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_bilinear_gather_nan_location_samples_zero():
    """An fp16 engine can produce NaN offsets: that must not become an out-of-range index."""
    locations = torch.full((1, 1, 2, 2), 0.5)
    locations[0, 0, 0, 0] = float("nan")

    got = bilinear_gather(torch.ones(1, 3, 4 * 6), locations, 4, 6)

    torch.testing.assert_close(got[..., 0], torch.zeros(1, 3, 1))
    torch.testing.assert_close(got[..., 1], torch.ones(1, 3, 1))


def test_bilinear_gather_exports_to_valid_onnx():
    """The unit tests above never run the exporter, and a static-shape graph once came out
    invalid (dangling Clip input) while the dynamic one was fine."""
    onnx = pytest.importorskip("onnx")  # the [export] extra
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    class Sampler(torch.nn.Module):
        def forward(self, value, locations):
            return bilinear_gather(value.flatten(2), locations, value.shape[2], value.shape[3])

    torch.manual_seed(0)
    value, locations = torch.randn(2, 4, 7, 11), torch.rand(2, 5, 3, 2) * 1.4 - 0.2
    program = torch.onnx.export(Sampler(), (value, locations), dynamo=True, opset_version=19)
    onnx.checker.check_model(program.model_proto, full_check=True)

    session = ort.InferenceSession(program.model_proto.SerializeToString())
    feeds = dict(zip([i.name for i in session.get_inputs()], (value.numpy(), locations.numpy())))
    want = F.grid_sample(value, 2 * locations - 1, align_corners=False)
    torch.testing.assert_close(
        torch.from_numpy(session.run(None, feeds)[0]), want, atol=1e-5, rtol=1e-5
    )


def test_exported_attention_core_matches_eager(monkeypatch):
    torch.manual_seed(0)
    bs, heads, c, lq = 2, 4, 8, 9
    shapes, points = [[8, 12], [4, 6], [2, 3]], [3, 6, 3]
    value = [torch.randn(bs, heads, c, h * w) for h, w in shapes]
    locations = torch.rand(bs, lq, heads, sum(points), 2) * 1.2 - 0.1
    weights = torch.rand(bs, lq, heads, sum(points)).softmax(-1)

    eager = deformable_attention_core_func_v2(value, shapes, locations, weights, points)
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    exported = deformable_attention_core_func_v2(value, shapes, locations, weights, points)

    torch.testing.assert_close(exported, eager, atol=1e-5, rtol=1e-5)
