import pytest
from omegaconf import OmegaConf

from dfine_seg.dl.train import _run_training


class FakeTrainer:
    def __init__(self, error=None):
        self.error = error
        self.is_main = True
        self.saved = []

    def train(self):
        if self.error:
            raise self.error

    def save_weights(self, filename):
        self.saved.append(filename)


def test_interrupted_training_saves_current_weights():
    trainer = FakeTrainer(KeyboardInterrupt())
    assert _run_training(trainer) is None
    assert trainer.saved == ["last.pt"]


def test_training_errors_propagate():
    trainer = FakeTrainer(RuntimeError("broken"))
    with pytest.raises(RuntimeError, match="broken"):
        _run_training(trainer)


def test_ddp_sampling_rejected_before_training_setup(monkeypatch):
    from dfine_seg.dl import train

    cfg = OmegaConf.create(
        {"train": {"ddp": {"enabled": True}, "rare_class_sampling": True, "batch_size": -1}}
    )
    monkeypatch.setattr(train, "is_dist_available_and_initialized", lambda: True)
    monkeypatch.setattr(train, "auto_batch_size", lambda *args: pytest.fail("Batch probe ran"))
    with pytest.raises(ValueError, match="rare_class_sampling.*DDP"):
        train.Trainer(cfg)


def test_regression_runner_sends_summary(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    from experiments import notify
    from scripts import regression_test

    sent = []
    monkeypatch.setattr(notify, "creds", lambda: ("test-token", "test-chat"))

    def post(url, *, json, timeout):
        sent.append(json)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(notify.requests, "post", post)
    monkeypatch.setattr(
        sys, "argv", ["regression_test.py", "--check-only", "--tasks", "sem", "--notify"]
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(regression_test, "run_dir_for", lambda *args: tmp_path)
    assert regression_test.main() == 1  # No run log: still notify the failure.
    assert len(sent) == 1
    assert sent[0]["chat_id"] == "test-chat"
    assert "regression FAILED" in sent[0]["text"] and "sem" in sent[0]["text"]
    assert "notification skipped" not in capsys.readouterr().out
