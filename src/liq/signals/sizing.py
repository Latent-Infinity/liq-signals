"""Signal sizing and order intent generation.

This module provides the bridge between signals (what to trade) and orders (how to trade).
It defines protocols for sizing strategies and utilities for converting signals to order intents.

Example:
    >>> from liq.signals import Signal, TargetWeightSizer, signal_to_order_intent
    >>> from liq.core import PortfolioState
    >>>
    >>> signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long", target_weight=0.1)
    >>> sizer = TargetWeightSizer(max_weight=0.2)
    >>> intent = sizer.size(signal, portfolio_state, current_price=50000.0)
    >>> # intent.quantity computed based on target_weight and portfolio equity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from liq.core import OrderSide, OrderType, PortfolioState

from liq.signals.exceptions import InsufficientDataError, InvalidSignalError

# Re-export Direction type for convenience
Direction = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class OrderIntent:
    """Intermediate representation bridging Signal to OrderRequest.

    OrderIntent contains all the information needed to create an OrderRequest,
    but remains broker-agnostic. It's produced by sizers and consumed by
    the order_intent_to_request converter.

    Attributes:
        symbol: Instrument symbol (e.g., "BTC_USDT")
        side: Order side (BUY or SELL)
        quantity: Order quantity (must be > 0)
        order_type: Type of order (MARKET, LIMIT, etc.)
        timestamp: When this intent was generated
        limit_price: Required for LIMIT orders
        stop_price: Required for STOP orders
        signal_metadata: Metadata from the source signal
        source_signal_id: Optional ID linking to the source signal
    """

    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    timestamp: datetime
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    signal_metadata: dict[str, Any] = field(default_factory=dict)
    source_signal_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the order intent."""
        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price required for LIMIT orders")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("stop_price required for STOP orders")


class SignalSizer(Protocol):
    """Protocol for signal sizing strategies.

    Implementations convert a Signal + context into an OrderIntent with
    a computed quantity. Different strategies may use fixed quantities,
    target weights, volatility-based sizing, etc.
    """

    def size(
        self,
        signal: Any,  # Signal type from liq.signals
        portfolio_state: PortfolioState,
        current_price: Decimal | float,
    ) -> OrderIntent | None:
        """Compute an OrderIntent from a signal.

        Args:
            signal: The signal to size
            portfolio_state: Current portfolio state for position/equity info
            current_price: Current market price for the symbol

        Returns:
            OrderIntent if the signal should generate an order, None otherwise
            (e.g., flat signal with no position returns None)

        Raises:
            SizingError: If sizing fails due to missing data or constraints
        """
        ...


def direction_to_side(
    direction: Direction, current_position_qty: Decimal
) -> OrderSide | None:
    """Convert signal direction to order side.

    Args:
        direction: Signal direction ("long", "short", "flat")
        current_position_qty: Current position quantity (positive=long, negative=short)

    Returns:
        OrderSide for the order, or None if no order needed (flat with no position)
    """
    if direction == "long":
        return OrderSide.BUY
    elif direction == "short":
        return OrderSide.SELL
    elif direction == "flat":
        # Flat signal closes existing position
        if current_position_qty > 0:
            return OrderSide.SELL
        elif current_position_qty < 0:
            return OrderSide.BUY
        else:
            return None  # No position to close
    else:
        raise InvalidSignalError(f"Invalid direction: {direction}")


def signal_to_order_intent(
    signal: Any,  # Signal from liq.signals
    portfolio_state: PortfolioState,
    current_price: Decimal | float,
    quantity: Decimal,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
    stop_price: Decimal | None = None,
) -> OrderIntent | None:
    """Convert a Signal to an OrderIntent.

    This is a low-level utility that creates an OrderIntent from a signal
    with a pre-computed quantity. For automatic quantity computation,
    use a SignalSizer implementation.

    Args:
        signal: The signal to convert
        portfolio_state: Current portfolio state
        current_price: Current market price (used for validation)
        quantity: Pre-computed order quantity
        order_type: Order type (default: MARKET)
        limit_price: Limit price for LIMIT orders
        stop_price: Stop price for STOP orders

    Returns:
        OrderIntent if an order should be placed, None if no action needed

    Raises:
        InvalidSignalError: If the signal is malformed
        InsufficientDataError: If required data is missing
    """
    if signal.timestamp is None:
        raise InvalidSignalError(
            "Signal missing timestamp", context={"symbol": signal.symbol}
        )

    # Get current position for this symbol
    position = portfolio_state.positions.get(signal.symbol)
    current_qty = position.quantity if position else Decimal("0")

    # Determine order side from direction
    side = direction_to_side(signal.direction, current_qty)
    if side is None:
        return None  # No order needed (e.g., flat with no position)

    # For flat signals, quantity is the current position size
    if signal.direction == "flat":
        quantity = abs(current_qty)

    if quantity <= 0:
        return None  # No order needed

    # Build signal metadata
    metadata: dict[str, Any] = dict(signal.metadata) if signal.metadata else {}
    metadata["source_direction"] = signal.direction
    metadata["source_strength"] = signal.strength

    return OrderIntent(
        symbol=signal.symbol,
        side=side,
        quantity=Decimal(str(quantity)),
        order_type=order_type,
        timestamp=signal.normalized_timestamp(),
        limit_price=limit_price,
        stop_price=stop_price,
        signal_metadata=metadata,
        source_signal_id=metadata.get("signal_id"),
    )


class FixedQuantitySizer:
    """Simple sizer that uses a fixed quantity per trade.

    This is the simplest sizing strategy, useful for testing or when
    position sizing is handled externally.

    Attributes:
        default_quantity: Fixed quantity for all orders
        order_type: Order type to use (default: MARKET)
    """

    def __init__(
        self,
        default_quantity: Decimal | float | str,
        order_type: OrderType = OrderType.MARKET,
    ) -> None:
        self.default_quantity = Decimal(str(default_quantity))
        self.order_type = order_type

    def size(
        self,
        signal: Any,
        portfolio_state: PortfolioState,
        current_price: Decimal | float,
    ) -> OrderIntent | None:
        """Size signal using fixed quantity.

        For flat signals, uses the current position quantity instead of default.
        """
        if signal.direction == "flat":
            position = portfolio_state.positions.get(signal.symbol)
            if not position or position.quantity == 0:
                return None
            qty = abs(position.quantity)
        else:
            qty = self.default_quantity

        return signal_to_order_intent(
            signal=signal,
            portfolio_state=portfolio_state,
            current_price=current_price,
            quantity=qty,
            order_type=self.order_type,
        )


class TargetWeightSizer:
    """Sizer that computes quantity from target portfolio weight.

    Uses signal.target_weight to determine the target position size as a
    percentage of portfolio equity. If target_weight is not set, falls back
    to signal.strength * max_weight.

    Attributes:
        max_weight: Maximum allowed weight per position (default: 1.0)
        min_order_value: Minimum order value to generate (filters noise)
        order_type: Order type to use (default: MARKET)
    """

    def __init__(
        self,
        max_weight: float = 1.0,
        min_order_value: Decimal | float | str = "0",
        order_type: OrderType = OrderType.MARKET,
    ) -> None:
        self.max_weight = max_weight
        self.min_order_value = Decimal(str(min_order_value))
        self.order_type = order_type

    def size(
        self,
        signal: Any,
        portfolio_state: PortfolioState,
        current_price: Decimal | float,
    ) -> OrderIntent | None:
        """Size signal based on target weight.

        Computes the quantity needed to reach the target portfolio weight,
        accounting for current position.
        """
        price = Decimal(str(current_price))
        if price <= 0:
            raise InsufficientDataError(
                "Invalid current price",
                context={"price": str(price), "symbol": signal.symbol},
            )

        equity = portfolio_state.equity
        if equity <= 0:
            raise InsufficientDataError(
                "Invalid portfolio equity",
                context={"equity": str(equity)},
            )

        # Get current position value
        position = portfolio_state.positions.get(signal.symbol)
        current_qty = position.quantity if position else Decimal("0")
        current_value = current_qty * price
        current_weight = float(current_value / equity)

        # Determine target weight
        if signal.direction == "flat":
            target_weight = 0.0
        elif signal.target_weight is not None:
            target_weight = min(signal.target_weight, self.max_weight)
        else:
            # Fallback: use strength * max_weight
            target_weight = signal.strength * self.max_weight

        # Apply sign based on direction
        if signal.direction == "short":
            target_weight = -abs(target_weight)
        elif signal.direction == "long":
            target_weight = abs(target_weight)

        # Compute weight delta
        weight_delta = target_weight - current_weight

        # Compute quantity from weight delta
        target_value_delta = Decimal(str(weight_delta)) * equity
        quantity = abs(target_value_delta / price)

        # Filter small orders
        order_value = quantity * price
        if order_value < self.min_order_value:
            return None

        # Determine side from weight delta
        if weight_delta > 0:
            side = OrderSide.BUY
        elif weight_delta < 0:
            side = OrderSide.SELL
        else:
            return None  # Already at target

        if quantity <= 0:
            return None

        # Build metadata
        metadata: dict[str, Any] = dict(signal.metadata) if signal.metadata else {}
        metadata["source_direction"] = signal.direction
        metadata["source_strength"] = signal.strength
        metadata["target_weight"] = target_weight
        metadata["current_weight"] = current_weight
        metadata["weight_delta"] = weight_delta

        return OrderIntent(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=self.order_type,
            timestamp=signal.normalized_timestamp(),
            signal_metadata=metadata,
            source_signal_id=metadata.get("signal_id"),
        )
