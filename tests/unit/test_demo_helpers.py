"""Demo helpers that are pure functions — the UI itself is not built here."""

import pytest

pytest.importorskip("gradio")  # the [demo] extra

from dfine_seg.demo import parse_names, parse_size  # noqa: E402


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
