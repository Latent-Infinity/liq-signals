import polars as pl
import pytest

from liq.signals.output import SignalOutput


def test_signal_output_validates_lengths() -> None:
    scores = pl.Series([0.1, 0.2])
    labels = pl.Series([1, 0])
    out = SignalOutput(scores=scores, labels=labels, metadata={"fold": 1})
    assert out.metadata["fold"] == 1


def test_signal_output_raises_on_mismatch() -> None:
    with pytest.raises(ValueError):
        SignalOutput(scores=pl.Series([0.1]), labels=pl.Series([1, 0]))


def test_signal_output_type_validation() -> None:
    with pytest.raises(TypeError):
        SignalOutput(scores=[0.1, 0.2])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SignalOutput(scores=pl.Series([0.1]), labels=[1, 0])  # type: ignore[arg-type]
