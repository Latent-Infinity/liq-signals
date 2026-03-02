"""Signal data structures and provider protocol for the LIQ stack.

This package provides:
- Signal: Core data structure for trading signals
- SignalProvider: Protocol for signal generation strategies
- Sizing: Signal → OrderIntent conversion with FixedQuantitySizer and TargetWeightSizer
- Processor: Full signal → OrderRequest pipeline via SignalProcessor

Example:
    >>> from liq.signals import Signal, SignalProcessor, TargetWeightSizer
    >>> from liq.core import PortfolioState
    >>>
    >>> # Create a signal
    >>> signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long", target_weight=0.1)
    >>>
    >>> # Process into an order
    >>> processor = SignalProcessor(sizer=TargetWeightSizer(max_weight=0.2))
    >>> order = processor.process_signal(signal, portfolio_state, current_price=50000)
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from liq.core.portfolio import PortfolioState

# Import sizing and processor components
from liq.signals.exceptions import (
    InsufficientDataError,
    InvalidSignalError,
    RiskConstraintError,
    SizingError,
)
from liq.signals.output import SignalOutput
from liq.signals.processor import (
    ProcessingResult,
    SignalProcessor,
    order_intent_to_request,
)
from liq.signals.sizing import (
    FixedQuantitySizer,
    OrderIntent,
    SignalSizer,
    TargetWeightSizer,
    direction_to_side,
    signal_to_order_intent,
)

Direction = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class Signal:
    """Model output describing what to trade (not how much).

    This keeps intelligence decoupled from sizing/execution. Risk/sizing layers
    turn signals into sized orders.
    """

    symbol: str
    timestamp: datetime
    direction: Direction
    strength: float = 1.0
    target_weight: float | None = None
    horizon: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_timestamp(self) -> datetime:
        """Return a timezone-aware UTC timestamp."""
        if (
            self.timestamp.tzinfo is None
            or self.timestamp.tzinfo.utcoffset(self.timestamp) is None
        ):
            return self.timestamp.replace(tzinfo=UTC)
        return self.timestamp.astimezone(UTC)


class SignalProvider(Protocol):
    """Any model/strategy implements this to integrate with the stack."""

    def generate_signals(
        self,
        data: Any | None = None,
        portfolio_state: PortfolioState | None = None,
    ) -> Iterable[Signal]: ...

    @property
    def required_history(self) -> int: ...

    @property
    def symbols(self) -> list[str]: ...

    @property
    def name(self) -> str: ...


class BaseSignalProvider:
    """Base implementation with defaults."""

    def __init__(self, name: str, symbols: Sequence[str], lookback: int = 0) -> None:
        self._name = name
        self._symbols = list(symbols)
        self._lookback = lookback

    @property
    def required_history(self) -> int:
        return self._lookback

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def name(self) -> str:
        return self._name


class FileSignalProvider(BaseSignalProvider):
    """Load signals from CSV or JSON for replay."""

    def __init__(self, filepath: Path, symbols: Sequence[str] | None = None) -> None:
        self.filepath = filepath
        self._cached: list[Signal] | None = None
        super().__init__(name="file_provider", symbols=symbols or [], lookback=0)

    def generate_signals(
        self, data: Any | None = None, portfolio_state: PortfolioState | None = None
    ) -> Iterable[Signal]:
        if self._cached is None:
            self._cached = list(self._load())
        if self.symbols:
            return [s for s in self._cached if s.symbol in self.symbols]
        return self._cached

    def _load(self) -> Iterable[Signal]:
        suffix = self.filepath.suffix.lower()
        if suffix == ".csv":
            yield from self._load_csv()
        elif suffix in {".json", ".jsonl"}:
            yield from self._load_json()
        else:
            raise ValueError(
                "Unsupported signals file format; use .csv or .json/.jsonl"
            )

    def _load_csv(self) -> Iterable[Signal]:
        with self.filepath.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row.get("timestamp", "") or row.get("ts", ""))
                if ts is None:
                    continue
                direction = str(row.get("direction", "flat")).lower()
                strength = float(row.get("strength", 1.0))
                target_weight = row.get("target_weight")
                horizon = row.get("horizon")
                yield Signal(
                    symbol=str(row.get("symbol", "")).upper(),
                    timestamp=ts,
                    direction=direction,  # type: ignore[arg-type]
                    strength=strength,
                    target_weight=float(target_weight)
                    if target_weight not in (None, "")
                    else None,
                    horizon=int(horizon) if horizon not in (None, "") else None,
                    metadata={
                        k: v
                        for k, v in row.items()
                        if k
                        not in {
                            "symbol",
                            "timestamp",
                            "ts",
                            "direction",
                            "strength",
                            "target_weight",
                            "horizon",
                        }
                    },
                )

    def _load_json(self) -> Iterable[Signal]:
        with self.filepath.open("r") as f:
            content = f.read().strip()
            if not content:
                return []
            if self.filepath.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in content.splitlines()]
            else:
                records = json.loads(content)
                if isinstance(records, dict):
                    records = records.get("signals", [])
            for row in records:
                ts = _parse_ts(row.get("timestamp") or row.get("ts") or "")
                if ts is None:
                    continue
                yield Signal(
                    symbol=str(row.get("symbol", "")).upper(),
                    timestamp=ts,
                    direction=str(row.get("direction", "flat")).lower(),  # type: ignore[arg-type]
                    strength=float(row.get("strength", 1.0)),
                    target_weight=row.get("target_weight"),
                    horizon=row.get("horizon"),
                    metadata={
                        k: v
                        for k, v in row.items()
                        if k
                        not in {
                            "symbol",
                            "timestamp",
                            "ts",
                            "direction",
                            "strength",
                            "target_weight",
                            "horizon",
                        }
                    },
                )


def _parse_ts(raw: str | datetime) -> datetime | None:
    if isinstance(raw, datetime):
        return raw
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


__all__ = [
    # Core signal types
    "Signal",
    "Direction",
    # Provider protocol and implementations
    "SignalProvider",
    "BaseSignalProvider",
    "FileSignalProvider",
    # Sizing
    "SignalSizer",
    "OrderIntent",
    "FixedQuantitySizer",
    "TargetWeightSizer",
    "signal_to_order_intent",
    "direction_to_side",
    # Processor
    "SignalProcessor",
    "ProcessingResult",
    "order_intent_to_request",
    "SignalOutput",
    # Exceptions
    "SizingError",
    "InsufficientDataError",
    "RiskConstraintError",
    "InvalidSignalError",
]
