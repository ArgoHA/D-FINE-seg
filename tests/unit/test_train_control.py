import pytest

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
