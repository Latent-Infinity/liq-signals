"""Exception hierarchy for signal processing and sizing.

This module defines exceptions for signal-to-order conversion failures,
enabling clear error handling throughout the pipeline.
"""

from __future__ import annotations

from typing import Any


class SizingError(Exception):
    """Base exception for signal sizing failures.

    Attributes:
        message: Human-readable error description
        signal_id: Optional identifier of the failing signal
        context: Additional context for debugging
    """

    def __init__(
        self,
        message: str,
        signal_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.signal_id = signal_id
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.signal_id:
            parts.append(f"signal_id={self.signal_id}")
        if self.context:
            parts.append(f"context={self.context}")
        return " | ".join(parts)


class InsufficientDataError(SizingError):
    """Raised when required market data is missing for sizing.

    Examples:
        - Missing price data for position sizing
        - Missing portfolio state
        - Missing required signal fields
    """

    pass


class RiskConstraintError(SizingError):
    """Raised when sizing would violate risk constraints.

    Examples:
        - Position size exceeds max allocation
        - Order would exceed gross leverage limit
        - Insufficient buying power
    """

    def __init__(
        self,
        message: str,
        constraint_name: str,
        constraint_value: Any,
        computed_value: Any,
        signal_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.constraint_name = constraint_name
        self.constraint_value = constraint_value
        self.computed_value = computed_value
        ctx = context or {}
        ctx.update({
            "constraint_name": constraint_name,
            "constraint_value": constraint_value,
            "computed_value": computed_value,
        })
        super().__init__(message, signal_id=signal_id, context=ctx)


class InvalidSignalError(SizingError):
    """Raised when a signal is malformed or invalid.

    Examples:
        - Invalid direction value
        - Missing required timestamp
        - Invalid strength value (outside [0, 1])
    """

    pass
