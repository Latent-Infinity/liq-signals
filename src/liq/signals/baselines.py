"""Baseline SignalProvider implementations for evaluation benchmarks.

Two baselines are provided:
    PassiveBaseline: equal-weight buy-and-hold (generates initial long signals)
    NaiveActiveBaseline: periodic equal-weight rebalance signals
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from liq.signals import BaseSignalProvider, Signal


class PassiveBaseline(BaseSignalProvider):
    """Equal-weight buy-and-hold baseline.

    Generates a single round of ``long`` signals with equal target weights
    across all symbols. Intended as a passive benchmark — once the initial
    allocation is placed, no further rebalancing occurs.
    """

    def __init__(
        self,
        symbols: list[str],
        name: str = "passive_equal_weight",
    ) -> None:
        super().__init__(name=name, symbols=symbols, lookback=0)

    def generate_signals(
        self,
        data: Any | None = None,
        portfolio_state: Any | None = None,
    ) -> Iterable[Signal]:
        ts: datetime = data if isinstance(data, datetime) else datetime.now(UTC)
        weight = 1.0 / len(self._symbols)
        for symbol in self._symbols:
            yield Signal(
                symbol=symbol,
                timestamp=ts,
                direction="long",
                strength=1.0,
                target_weight=weight,
            )


class NaiveActiveBaseline(BaseSignalProvider):
    """Naive equal-weight rebalance baseline.

    On every call to ``generate_signals``, emits ``long`` signals with
    equal target weights for all symbols.  When fed to a
    ``TargetWeightSizer``, this produces orders that rebalance the
    portfolio back to equal weights — trimming overweight positions and
    adding to underweight ones.

    The optional ``rebalance_threshold`` controls the minimum weight
    drift before a rebalance signal is emitted (not enforced here;
    downstream sizers handle minimum-delta filtering).
    """

    def __init__(
        self,
        symbols: list[str],
        name: str = "naive_active_rebalance",
        rebalance_threshold: float = 0.01,
    ) -> None:
        super().__init__(name=name, symbols=symbols, lookback=0)
        self._rebalance_threshold = rebalance_threshold

    def generate_signals(
        self,
        data: Any | None = None,
        portfolio_state: Any | None = None,
    ) -> Iterable[Signal]:
        ts: datetime = data if isinstance(data, datetime) else datetime.now(UTC)
        weight = 1.0 / len(self._symbols)
        for symbol in self._symbols:
            yield Signal(
                symbol=symbol,
                timestamp=ts,
                direction="long",
                strength=1.0,
                target_weight=weight,
                metadata={"baseline_type": self._name},
            )
