"""Signal processing and order generation.

This module provides the SignalProcessor class that orchestrates the full
signal → order pipeline, combining sizing strategies with order conversion.

Example:
    >>> from liq.signals import Signal, SignalProcessor, TargetWeightSizer
    >>> from liq.core import PortfolioState
    >>>
    >>> processor = SignalProcessor(sizer=TargetWeightSizer(max_weight=0.2))
    >>> signals = [Signal(symbol="BTC_USDT", timestamp=now, direction="long", target_weight=0.1)]
    >>> orders = processor.process_signals(signals, portfolio_state, prices={"BTC_USDT": 50000})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from liq.core import OrderRequest, PortfolioState

from liq.signals.exceptions import InvalidSignalError, SizingError
from liq.signals.sizing import OrderIntent, SignalSizer

logger = logging.getLogger(__name__)


def order_intent_to_request(
    intent: OrderIntent,
    strategy_id: str | None = None,
    confidence: float | None = None,
) -> OrderRequest:
    """Convert an OrderIntent to a liq-core OrderRequest.

    This is the final step in the signal → order pipeline, creating a
    fully-validated OrderRequest ready for submission to liq-sim or a broker.

    Args:
        intent: The OrderIntent to convert
        strategy_id: Optional strategy identifier for the order
        confidence: Optional confidence score [0, 1]

    Returns:
        A valid OrderRequest ready for execution

    Example:
        >>> intent = OrderIntent(symbol="BTC_USDT", side=OrderSide.BUY, ...)
        >>> order = order_intent_to_request(intent, strategy_id="momentum_v1")
    """
    return OrderRequest(
        symbol=intent.symbol,
        side=intent.side,
        order_type=intent.order_type,
        quantity=intent.quantity,
        limit_price=intent.limit_price,
        stop_price=intent.stop_price,
        timestamp=intent.timestamp,
        strategy_id=strategy_id,
        confidence=confidence,
        metadata=intent.signal_metadata,
    )


@dataclass
class ProcessingResult:
    """Result of processing a batch of signals.

    Attributes:
        orders: Successfully generated OrderRequests
        skipped: Signals that were skipped (e.g., flat with no position)
        errors: Signals that failed to process with their error messages
    """

    orders: list[OrderRequest] = field(default_factory=list)
    skipped: list[Any] = field(default_factory=list)  # Signal type
    errors: list[tuple[Any, str]] = field(default_factory=list)  # (Signal, error_msg)


class SignalProcessor:
    """Orchestrates signal-to-order conversion.

    Combines a SignalSizer with order conversion to provide a complete
    signal processing pipeline. Handles errors gracefully and logs
    processing decisions.

    Attributes:
        sizer: The sizing strategy to use
        strategy_id: Optional strategy identifier for generated orders
        default_confidence: Default confidence if not in signal metadata

    Example:
        >>> processor = SignalProcessor(sizer=TargetWeightSizer())
        >>> result = processor.process_signals(signals, portfolio, prices)
        >>> print(f"Generated {len(result.orders)} orders, skipped {len(result.skipped)}")
    """

    def __init__(
        self,
        sizer: SignalSizer,
        strategy_id: str | None = None,
        default_confidence: float | None = None,
    ) -> None:
        self.sizer = sizer
        self.strategy_id = strategy_id
        self.default_confidence = default_confidence

    def process_signal(
        self,
        signal: Any,  # Signal from liq.signals
        portfolio_state: PortfolioState,
        current_price: Decimal | float,
    ) -> OrderRequest | None:
        """Process a single signal into an OrderRequest.

        Args:
            signal: The signal to process
            portfolio_state: Current portfolio state
            current_price: Current price for the signal's symbol

        Returns:
            OrderRequest if the signal produces an order, None otherwise

        Raises:
            SizingError: If signal processing fails
        """
        logger.debug(
            "Processing signal",
            extra={
                "symbol": signal.symbol,
                "direction": signal.direction,
                "strength": signal.strength,
            },
        )

        intent = self.sizer.size(signal, portfolio_state, current_price)

        if intent is None:
            logger.debug(
                "Signal skipped (no order needed)",
                extra={"symbol": signal.symbol, "direction": signal.direction},
            )
            return None

        # Extract confidence from signal metadata or use default
        confidence = signal.metadata.get("confidence") if signal.metadata else None
        if confidence is None:
            confidence = self.default_confidence

        order = order_intent_to_request(
            intent,
            strategy_id=self.strategy_id,
            confidence=confidence,
        )

        logger.info(
            "Order generated",
            extra={
                "symbol": order.symbol,
                "side": str(order.side),
                "quantity": str(order.quantity),
                "order_type": str(order.order_type),
            },
        )

        return order

    def process_signals(
        self,
        signals: list[Any],  # list[Signal]
        portfolio_state: PortfolioState,
        prices: dict[str, Decimal | float],
    ) -> ProcessingResult:
        """Process multiple signals into OrderRequests.

        Args:
            signals: List of signals to process
            portfolio_state: Current portfolio state
            prices: Dictionary mapping symbol to current price

        Returns:
            ProcessingResult containing orders, skipped signals, and errors

        Example:
            >>> result = processor.process_signals(
            ...     signals=[sig1, sig2, sig3],
            ...     portfolio_state=portfolio,
            ...     prices={"BTC_USDT": 50000, "ETH_USDT": 3000},
            ... )
        """
        result = ProcessingResult()

        for signal in signals:
            try:
                price = prices.get(signal.symbol)
                if price is None:
                    raise InvalidSignalError(
                        f"No price available for {signal.symbol}",
                        context={"symbol": signal.symbol},
                    )

                order = self.process_signal(signal, portfolio_state, price)

                if order is not None:
                    result.orders.append(order)
                else:
                    result.skipped.append(signal)

            except SizingError as e:
                logger.warning(
                    "Signal processing failed",
                    extra={
                        "symbol": signal.symbol,
                        "error": str(e),
                    },
                )
                result.errors.append((signal, str(e)))

        logger.info(
            "Batch processing complete",
            extra={
                "total_signals": len(signals),
                "orders_generated": len(result.orders),
                "skipped": len(result.skipped),
                "errors": len(result.errors),
            },
        )

        return result
