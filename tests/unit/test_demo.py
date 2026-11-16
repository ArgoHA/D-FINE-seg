"""Demo server: the pure helpers plus the HTTP API, driven by a stub model (no weights)."""

import http.client
import json
import threading
import time

import cv2
import numpy as np
import pytest
import torch

from dfine_seg.app import demo
from dfine_seg.app.demo import parse_names, parse_size
from dfine_seg.viz import Visualizer


@pytest.mark.parametrize(
    "text, expected",
    [("640", (640, 640)), ("1024x2048", (1024, 2048)), ("1024, 2048", (1024, 2048))],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "0", "-5", "12x", "1x2x3"])
def test_parse_size_rejects_junk(text):
    """`12x` must not silently read as 12 — a trailing separator is a half-typed size."""
    with pytest.raises(ValueError):
        parse_size(text)


def test_parse_names_ids_and_order():
    assert parse_names("person, car") == {0: "person", 1: "car"}
    assert parse_names("3: dog") == {3: "dog"}
    assert parse_names("") is None


class StubModel:
    """One box in the top-left corner, whatever the input."""

    def __init__(self):
        self.seen_conf = []

    def __call__(self, img, conf_thresh=None):
        self.seen_conf.append(conf_thresh)
        return [
            {
                "labels": torch.tensor([1]),
                "boxes": torch.tensor([[2.0, 2.0, 20.0, 20.0]]),
                "scores": torch.tensor([0.9]),
            }
        ]


@pytest.fixture
def server(monkeypatch, tmp_path):
    stub = StubModel()
    monkeypatch.setattr(demo, "CURRENT", demo.Loaded(stub, Visualizer(2), {"spec": "stub"}))
    monkeypatch.setattr(demo, "WORK_DIR", tmp_path)
    monkeypatch.setattr(demo, "JOBS", {})
    srv = demo.make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def request(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=30)
        conn.request(method, path, body=body, headers=headers or {})
        res = conn.getresponse()
        return res, res.read()

    request.stub = stub
    yield request
    srv.shutdown()
    srv.server_close()


POST = {"X-DFine": "1"}


def _jpeg(h=48, w=64):
    return cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))[1].tobytes()


def test_serves_the_page(server):
    for path in demo.STATIC:
        res, body = server("GET", path)
        assert res.status == 200 and body, path
    res, body = server("GET", "/api/state")
    state = json.loads(body)
    assert state["model"]["spec"] == "stub"
    assert {"key": "s (sem_seg)", "size": "s", "task": "sem_seg"} in state["presets"]


def test_post_needs_the_custom_header(server):
    """Without it a cross-site form post could drive the API (it loads local files)."""
    res, _ = server("POST", "/api/image", _jpeg())
    assert res.status == 403


def test_image_roundtrip(server):
    res, body = server("POST", "/api/image?conf=0.3&minimize=1", _jpeg(), POST)
    assert res.status == 200 and res.getheader("Content-Type") == "image/jpeg"
    assert res.getheader("X-Summary") == "1 object"
    assert float(res.getheader("X-Infer-Ms")) >= 0
    out = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (48, 64, 3)
    assert server.stub.seen_conf == [0.3]


def test_bad_inputs_report_errors(server):
    res, body = server("POST", "/api/image", b"not an image", POST)
    assert res.status == 400 and "decode" in json.loads(body)["error"]
    res, _ = server("POST", "/api/upload?name=evil.sh", b"x", POST)
    assert res.status == 400
    res, body = server("POST", "/api/load", json.dumps({"model": "/no/such.pt"}), POST)
    assert res.status == 422 and "FileNotFoundError" in json.loads(body)["error"]
    assert demo.CURRENT.info["spec"] == "stub"  # a failed load keeps the working model


def test_upload_keeps_the_name_inside_the_work_dir(server, tmp_path):
    res, body = server("POST", "/api/upload?name=../../x%20y.pt", b"weights", POST)
    assert res.status == 200
    assert json.loads(body)["path"] == str(tmp_path / "weights" / "x_y.pt")


def test_video_job_and_range_requests(server, tmp_path):
    src = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(6):
        writer.write(np.full((48, 64, 3), 100, np.uint8))
    writer.release()

    def wait(job):
        for _ in range(100):
            state = json.loads(server("GET", f"/api/video/{job}")[1])
            if state["status"] in ("done", "error"):
                return state
            time.sleep(0.1)

    _, body = server("POST", "/api/video?stride=2", src.read_bytes(), POST)
    first = json.loads(body)["job"]
    state = wait(first)
    assert state["status"] == "done", state
    assert state["frames"] == 6
    assert server.stub.seen_conf == [0.5] * 3  # stride 2: frames 0, 2, 4

    # "Process again": new settings on the kept upload, no body; the old result is dropped
    _, body = server("POST", f"/api/video/{first}/rerun?stride=3&conf=0.7", None, POST)
    job = json.loads(body)["job"]
    assert wait(job)["status"] == "done"
    assert server.stub.seen_conf[3:] == [0.7] * 2  # stride 3: frames 0, 3
    assert server("GET", f"/api/video/{first}")[0].status == 404

    res, full = server("GET", f"/api/video/{job}/result")
    assert res.status == 200 and len(full) > 0
    res, part = server("GET", f"/api/video/{job}/result", headers={"Range": "bytes=10-19"})
    assert res.status == 206 and part == full[10:20]
    assert res.getheader("Content-Range") == f"bytes 10-19/{len(full)}"
    res, tail = server("GET", f"/api/video/{job}/result", headers={"Range": "bytes=-5"})
    assert res.status == 206 and tail == full[-5:]
    for bad in ("bytes=-0", "bytes=20-10", f"bytes={len(full)}-"):
        res, body = server("GET", f"/api/video/{job}/result", headers={"Range": bad})
        assert res.status == 416 and body == b"", bad
        assert res.getheader("Content-Range") == f"bytes */{len(full)}"


def test_rejects_foreign_host_header(server):
    """DNS rebinding: an attacker domain resolved to 127.0.0.1 is same-origin to its page."""
    evil = {"Host": "evil.example:7860"}
    assert server("GET", "/api/state", headers=evil)[0].status == 403
    assert server("POST", "/api/image", _jpeg(), {**POST, **evil})[0].status == 403
    for ok in ("localhost:7860", "127.0.0.1", "192.168.1.5:7860", "[::1]:7860"):
        assert server("GET", "/api/state", headers={"Host": ok})[0].status == 200, ok


def test_sam3_prompts_set_once_per_change(monkeypatch):
    """Setting prompts drops SAM3's cached text embeddings, so a run must not reset them."""

    class Sam:
        prompts, sets = ["old"], 0

        def __setattr__(self, k, v):
            if k == "prompts":
                type(self).sets += 1
            object.__setattr__(self, k, v)

        parse_prompts = staticmethod(lambda text: [t.strip() for t in text.split(",")])

        def __call__(self, img, conf_thresh):
            return [{"labels": [], "boxes": [], "scores": []}]

    sam = Sam()
    monkeypatch.setattr(demo, "SAM3_AVAILABLE", True)
    monkeypatch.setattr(demo, "_get_sam_model", lambda: sam)
    predict, _ = demo.Run(backend="sam3", prompt="person, car").predictor()
    for _ in range(5):
        predict(np.zeros((8, 8, 3), np.uint8))
    assert sam.prompts == ["person", "car"] and Sam.sets == 1
