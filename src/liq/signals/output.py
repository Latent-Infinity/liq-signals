"""Signal output contract for rolling strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass(frozen=True)
class SignalOutput:
    """Model outputs for a symbol/timeframe."""

    scores: pl.Series
    labels: pl.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scores, pl.Series):
            raise TypeError("scores must be a polars Series")
        if self.labels is not None and not isinstance(self.labels, pl.Series):
            raise TypeError("labels must be a polars Series when provided")
        if self.labels is not None and self.labels.len() != self.scores.len():
            raise ValueError("scores and labels must have equal length when labels provided")
