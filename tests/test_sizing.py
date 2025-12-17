"""Tests for signal sizing module.

Following TDD: Tests cover all sizing functionality including:
- OrderIntent creation and validation
- Direction to side conversion
- signal_to_order_intent utility
- FixedQuantitySizer implementation
- TargetWeightSizer implementation
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import OrderSide, OrderType, PortfolioState, Position

from liq.signals import Signal
from liq.signals.exceptions import (
    InsufficientDataError,
    InvalidSignalError,
    RiskConstraintError,
    SizingError,
)
from liq.signals.sizing import (
    FixedQuantitySizer,
    OrderIntent,
    TargetWeightSizer,
    direction_to_side,
    signal_to_order_intent,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def now() -> datetime:
    """Current UTC timestamp."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)


@pytest.fixture
def empty_portfolio() -> PortfolioState:
    """Portfolio with no positions."""
    return PortfolioState(
        cash=Decimal("100000"),
        equity=Decimal("100000"),
        positions={},
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
    )


@pytest.fixture
def portfolio_with_long(now: datetime) -> PortfolioState:
    """Portfolio with a long position."""
    return PortfolioState(
        cash=Decimal("50000"),
        equity=Decimal("100000"),
        positions={
            "BTC_USDT": Position(
                symbol="BTC_USDT",
                quantity=Decimal("1.0"),
                average_price=Decimal("50000"),
                realized_pnl=Decimal("0"),
                timestamp=now,
            )
        },
        timestamp=now,
    )


@pytest.fixture
def portfolio_with_short(now: datetime) -> PortfolioState:
    """Portfolio with a short position."""
    return PortfolioState(
        cash=Decimal("150000"),
        equity=Decimal("100000"),
        positions={
            "BTC_USDT": Position(
                symbol="BTC_USDT",
                quantity=Decimal("-1.0"),
                average_price=Decimal("50000"),
                realized_pnl=Decimal("0"),
                timestamp=now,
            )
        },
        timestamp=now,
    )


# ============================================================================
# Tests for Exceptions
# ============================================================================


class TestExceptions:
    """Tests for exception hierarchy."""

    def test_sizing_error_basic(self) -> None:
        """SizingError should store message and context."""
        err = SizingError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.signal_id is None
        assert err.context == {}

    def test_sizing_error_with_context(self) -> None:
        """SizingError should include signal_id and context in string."""
        err = SizingError(
            "Failed to size",
            signal_id="sig-123",
            context={"symbol": "BTC_USDT"},
        )
        assert "Failed to size" in str(err)
        assert "sig-123" in str(err)
        assert "BTC_USDT" in str(err)

    def test_insufficient_data_error(self) -> None:
        """InsufficientDataError should inherit from SizingError."""
        err = InsufficientDataError("Missing price data")
        assert isinstance(err, SizingError)

    def test_risk_constraint_error(self) -> None:
        """RiskConstraintError should include constraint details."""
        err = RiskConstraintError(
            "Position too large",
            constraint_name="max_position_pct",
            constraint_value=0.1,
            computed_value=0.15,
        )
        assert isinstance(err, SizingError)
        assert err.constraint_name == "max_position_pct"
        assert "max_position_pct" in str(err)

    def test_invalid_signal_error(self) -> None:
        """InvalidSignalError should inherit from SizingError."""
        err = InvalidSignalError("Invalid direction")
        assert isinstance(err, SizingError)


# ============================================================================
# Tests for OrderIntent
# ============================================================================


class TestOrderIntent:
    """Tests for OrderIntent dataclass."""

    def test_create_market_order_intent(self, now: datetime) -> None:
        """Should create a valid market order intent."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
            timestamp=now,
        )

        assert intent.symbol == "BTC_USDT"
        assert intent.side == OrderSide.BUY
        assert intent.quantity == Decimal("0.5")
        assert intent.order_type == OrderType.MARKET

    def test_create_limit_order_intent(self, now: datetime) -> None:
        """Should create a valid limit order intent."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.LIMIT,
            timestamp=now,
            limit_price=Decimal("45000"),
        )

        assert intent.limit_price == Decimal("45000")

    def test_limit_order_requires_price(self, now: datetime) -> None:
        """LIMIT order without limit_price should raise ValueError."""
        with pytest.raises(ValueError, match="limit_price required"):
            OrderIntent(
                symbol="BTC_USDT",
                side=OrderSide.BUY,
                quantity=Decimal("0.5"),
                order_type=OrderType.LIMIT,
                timestamp=now,
            )

    def test_stop_order_requires_price(self, now: datetime) -> None:
        """STOP order without stop_price should raise ValueError."""
        with pytest.raises(ValueError, match="stop_price required"):
            OrderIntent(
                symbol="BTC_USDT",
                side=OrderSide.SELL,
                quantity=Decimal("0.5"),
                order_type=OrderType.STOP,
                timestamp=now,
            )

    def test_quantity_must_be_positive(self, now: datetime) -> None:
        """Quantity must be > 0."""
        with pytest.raises(ValueError, match="quantity must be > 0"):
            OrderIntent(
                symbol="BTC_USDT",
                side=OrderSide.BUY,
                quantity=Decimal("0"),
                order_type=OrderType.MARKET,
                timestamp=now,
            )

    def test_order_intent_is_frozen(self, now: datetime) -> None:
        """OrderIntent should be immutable."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
            timestamp=now,
        )

        with pytest.raises((TypeError, AttributeError)):  # FrozenInstanceError
            intent.quantity = Decimal("1.0")  # type: ignore[misc]

    def test_order_intent_with_metadata(self, now: datetime) -> None:
        """Should store signal metadata."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
            timestamp=now,
            signal_metadata={"model": "lstm", "confidence": 0.95},
            source_signal_id="sig-abc",
        )

        assert intent.signal_metadata["model"] == "lstm"
        assert intent.source_signal_id == "sig-abc"


# ============================================================================
# Tests for direction_to_side
# ============================================================================


class TestDirectionToSide:
    """Tests for direction_to_side utility."""

    def test_long_returns_buy(self) -> None:
        """Long signal should return BUY."""
        side = direction_to_side("long", Decimal("0"))
        assert side == OrderSide.BUY

    def test_short_returns_sell(self) -> None:
        """Short signal should return SELL."""
        side = direction_to_side("short", Decimal("0"))
        assert side == OrderSide.SELL

    def test_flat_with_long_position_returns_sell(self) -> None:
        """Flat signal with long position should return SELL to close."""
        side = direction_to_side("flat", Decimal("1.0"))
        assert side == OrderSide.SELL

    def test_flat_with_short_position_returns_buy(self) -> None:
        """Flat signal with short position should return BUY to close."""
        side = direction_to_side("flat", Decimal("-1.0"))
        assert side == OrderSide.BUY

    def test_flat_with_no_position_returns_none(self) -> None:
        """Flat signal with no position should return None."""
        side = direction_to_side("flat", Decimal("0"))
        assert side is None

    def test_invalid_direction_raises_error(self) -> None:
        """Invalid direction should raise InvalidSignalError."""
        with pytest.raises(InvalidSignalError, match="Invalid direction"):
            direction_to_side("invalid", Decimal("0"))  # type: ignore[arg-type]


# ============================================================================
# Tests for signal_to_order_intent
# ============================================================================


class TestSignalToOrderIntent:
    """Tests for signal_to_order_intent utility."""

    def test_long_signal_creates_buy_intent(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Long signal should create BUY intent."""
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            strength=0.8,
        )

        intent = signal_to_order_intent(
            signal=signal,
            portfolio_state=empty_portfolio,
            current_price=50000,
            quantity=Decimal("0.5"),
        )

        assert intent is not None
        assert intent.side == OrderSide.BUY
        assert intent.quantity == Decimal("0.5")
        assert intent.symbol == "BTC_USDT"

    def test_short_signal_creates_sell_intent(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Short signal should create SELL intent."""
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="short",
            strength=0.8,
        )

        intent = signal_to_order_intent(
            signal=signal,
            portfolio_state=empty_portfolio,
            current_price=50000,
            quantity=Decimal("0.5"),
        )

        assert intent is not None
        assert intent.side == OrderSide.SELL

    def test_flat_signal_closes_long_position(
        self, now: datetime, portfolio_with_long: PortfolioState
    ) -> None:
        """Flat signal should close existing long position."""
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="flat",
        )

        intent = signal_to_order_intent(
            signal=signal,
            portfolio_state=portfolio_with_long,
            current_price=50000,
            quantity=Decimal("999"),  # Should be ignored for flat
        )

        assert intent is not None
        assert intent.side == OrderSide.SELL
        assert intent.quantity == Decimal("1.0")  # Current position size

    def test_flat_signal_closes_short_position(
        self, now: datetime, portfolio_with_short: PortfolioState
    ) -> None:
        """Flat signal should close existing short position."""
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="flat",
        )

        intent = signal_to_order_intent(
            signal=signal,
            portfolio_state=portfolio_with_short,
            current_price=50000,
            quantity=Decimal("999"),
        )

        assert intent is not None
        assert intent.side == OrderSide.BUY
        assert intent.quantity == Decimal("1.0")

    def test_flat_signal_no_position_returns_none(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Flat signal with no position should return None."""
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="flat",
        )

        intent = signal_to_order_intent(
            signal=signal,
            portfolio_state=empty_portfolio,
            current_price=50000,
            quantity=Decimal("0.5"),
        )

        assert intent is None

    def test_signal_metadata_preserved(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Signal metadata should be preserved in intent."""
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            strength=0.8,
            metadata={"model": "transformer", "version": "1.0"},
        )

        intent = signal_to_order_intent(
            signal=signal,
            portfolio_state=empty_portfolio,
            current_price=50000,
            quantity=Decimal("0.5"),
        )

        assert intent is not None
        assert intent.signal_metadata["model"] == "transformer"
        assert intent.signal_metadata["source_strength"] == 0.8
        assert intent.signal_metadata["source_direction"] == "long"

    def test_zero_quantity_returns_none(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Zero quantity should return None."""
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
        )

        intent = signal_to_order_intent(
            signal=signal,
            portfolio_state=empty_portfolio,
            current_price=50000,
            quantity=Decimal("0"),
        )

        assert intent is None


# ============================================================================
# Tests for FixedQuantitySizer
# ============================================================================


class TestFixedQuantitySizer:
    """Tests for FixedQuantitySizer implementation."""

    def test_long_signal_uses_fixed_quantity(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Long signal should use fixed quantity."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long")

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.quantity == Decimal("0.5")
        assert intent.side == OrderSide.BUY

    def test_short_signal_uses_fixed_quantity(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Short signal should use fixed quantity."""
        sizer = FixedQuantitySizer(default_quantity=1.0)
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="short")

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.quantity == Decimal("1.0")
        assert intent.side == OrderSide.SELL

    def test_flat_signal_uses_position_quantity(
        self, now: datetime, portfolio_with_long: PortfolioState
    ) -> None:
        """Flat signal should use current position quantity, not fixed."""
        sizer = FixedQuantitySizer(default_quantity="0.1")  # Should be ignored
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="flat")

        intent = sizer.size(signal, portfolio_with_long, current_price=50000)

        assert intent is not None
        assert intent.quantity == Decimal("1.0")  # Position size
        assert intent.side == OrderSide.SELL

    def test_flat_no_position_returns_none(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Flat signal with no position should return None."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="flat")

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is None

    def test_custom_order_type(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should support custom order types."""
        sizer = FixedQuantitySizer(
            default_quantity="0.5",
            order_type=OrderType.MARKET,
        )
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long")

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.order_type == OrderType.MARKET


# ============================================================================
# Tests for TargetWeightSizer
# ============================================================================


class TestTargetWeightSizer:
    """Tests for TargetWeightSizer implementation."""

    def test_target_weight_computes_quantity(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should compute quantity from target_weight."""
        # 10% of $100k equity at $50k/BTC = 0.2 BTC
        sizer = TargetWeightSizer()
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            target_weight=0.1,
        )

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.quantity == Decimal("0.2")
        assert intent.side == OrderSide.BUY

    def test_fallback_to_strength_when_no_target_weight(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should use strength * max_weight when target_weight is None."""
        # strength=0.5 * max_weight=0.2 = 10% weight
        # 10% of $100k at $50k = 0.2 BTC
        sizer = TargetWeightSizer(max_weight=0.2)
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            strength=0.5,
        )

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.quantity == Decimal("0.2")

    def test_position_adjustment_buy_more(
        self, now: datetime, portfolio_with_long: PortfolioState
    ) -> None:
        """Should compute delta when already have position."""
        # Current: 1 BTC at $50k = $50k = 50% weight
        # Target: 60% weight = $60k = 1.2 BTC
        # Delta: 0.2 BTC buy
        sizer = TargetWeightSizer()
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            target_weight=0.6,
        )

        intent = sizer.size(signal, portfolio_with_long, current_price=50000)

        assert intent is not None
        assert intent.side == OrderSide.BUY
        # Use approximate comparison due to floating point arithmetic
        assert abs(intent.quantity - Decimal("0.2")) < Decimal("0.0001")

    def test_position_adjustment_reduce(
        self, now: datetime, portfolio_with_long: PortfolioState
    ) -> None:
        """Should sell to reduce position."""
        # Current: 1 BTC at $50k = 50% weight
        # Target: 30% weight = $30k = 0.6 BTC
        # Delta: -0.4 BTC = sell 0.4 BTC
        sizer = TargetWeightSizer()
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            target_weight=0.3,
        )

        intent = sizer.size(signal, portfolio_with_long, current_price=50000)

        assert intent is not None
        assert intent.side == OrderSide.SELL
        assert intent.quantity == Decimal("0.4")

    def test_flat_signal_closes_position(
        self, now: datetime, portfolio_with_long: PortfolioState
    ) -> None:
        """Flat signal should target 0% weight."""
        # Current: 1 BTC = 50% weight
        # Target: 0% = sell all 1 BTC
        sizer = TargetWeightSizer()
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="flat")

        intent = sizer.size(signal, portfolio_with_long, current_price=50000)

        assert intent is not None
        assert intent.side == OrderSide.SELL
        assert intent.quantity == Decimal("1.0")

    def test_short_signal_negative_weight(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Short signal should produce negative target weight."""
        # 10% short of $100k at $50k = -0.2 BTC
        sizer = TargetWeightSizer()
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="short",
            target_weight=0.1,
        )

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.side == OrderSide.SELL
        assert intent.quantity == Decimal("0.2")

    def test_max_weight_caps_target(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should cap target_weight at max_weight."""
        # target_weight=0.5 but max_weight=0.2 -> use 0.2
        # 20% of $100k at $50k = 0.4 BTC
        sizer = TargetWeightSizer(max_weight=0.2)
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            target_weight=0.5,
        )

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.quantity == Decimal("0.4")

    def test_already_at_target_returns_none(
        self, now: datetime, portfolio_with_long: PortfolioState
    ) -> None:
        """Should return None if already at target weight."""
        # Current: 1 BTC at $50k = 50% weight
        # Target: 50% weight -> no action needed
        sizer = TargetWeightSizer()
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            target_weight=0.5,
        )

        intent = sizer.size(signal, portfolio_with_long, current_price=50000)

        assert intent is None

    def test_min_order_value_filters_small_orders(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should filter orders below min_order_value."""
        # 1% of $100k at $50k = 0.02 BTC = $1000 value
        # But min_order_value=$5000 -> filtered out
        sizer = TargetWeightSizer(min_order_value="5000")
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            target_weight=0.01,
        )

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is None

    def test_invalid_price_raises_error(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should raise error for invalid price."""
        sizer = TargetWeightSizer()
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long")

        with pytest.raises(InsufficientDataError, match="Invalid current price"):
            sizer.size(signal, empty_portfolio, current_price=0)

    def test_zero_equity_raises_error(self, now: datetime) -> None:
        """Should raise error for zero equity."""
        portfolio = PortfolioState(
            cash=Decimal("0"),
            equity=Decimal("0"),
            positions={},
            timestamp=now,
        )
        sizer = TargetWeightSizer()
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long")

        with pytest.raises(InsufficientDataError, match="Invalid portfolio equity"):
            sizer.size(signal, portfolio, current_price=50000)

    def test_metadata_includes_weight_info(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should include weight calculation info in metadata."""
        sizer = TargetWeightSizer()
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            target_weight=0.1,
            metadata={"model": "test"},
        )

        intent = sizer.size(signal, empty_portfolio, current_price=50000)

        assert intent is not None
        assert intent.signal_metadata["target_weight"] == 0.1
        assert intent.signal_metadata["current_weight"] == 0.0
        assert intent.signal_metadata["weight_delta"] == 0.1
        assert intent.signal_metadata["model"] == "test"
