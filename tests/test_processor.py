"""Tests for signal processor module.

Following TDD: Tests cover order_intent_to_request and SignalProcessor.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.signals import Signal
from liq.signals.processor import (
    ProcessingResult,
    SignalProcessor,
    order_intent_to_request,
)
from liq.signals.sizing import FixedQuantitySizer, OrderIntent

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def now() -> datetime:
    """Current UTC timestamp."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)


@pytest.fixture
def empty_portfolio(now: datetime) -> PortfolioState:
    """Portfolio with no positions."""
    return PortfolioState(
        cash=Decimal("100000"),
        equity=Decimal("100000"),
        positions={},
        timestamp=now,
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


# ============================================================================
# Tests for order_intent_to_request
# ============================================================================


class TestOrderIntentToRequest:
    """Tests for order_intent_to_request converter."""

    def test_market_order_conversion(self, now: datetime) -> None:
        """Should convert market order intent to OrderRequest."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
            timestamp=now,
        )

        order = order_intent_to_request(intent)

        assert isinstance(order, OrderRequest)
        assert order.symbol == "BTC_USDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("0.5")
        assert order.order_type == OrderType.MARKET
        assert order.timestamp == now

    def test_limit_order_conversion(self, now: datetime) -> None:
        """Should convert limit order intent with price."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.LIMIT,
            timestamp=now,
            limit_price=Decimal("45000"),
        )

        order = order_intent_to_request(intent)

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("45000")

    def test_metadata_preserved(self, now: datetime) -> None:
        """Should preserve signal metadata in order."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
            timestamp=now,
            signal_metadata={"model": "lstm", "version": "1.0"},
        )

        order = order_intent_to_request(intent)

        assert order.metadata is not None
        assert order.metadata["model"] == "lstm"
        assert order.metadata["version"] == "1.0"

    def test_strategy_id_passed(self, now: datetime) -> None:
        """Should include strategy_id in order."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
            timestamp=now,
        )

        order = order_intent_to_request(intent, strategy_id="momentum_v2")

        assert order.strategy_id == "momentum_v2"

    def test_confidence_passed(self, now: datetime) -> None:
        """Should include confidence in order."""
        intent = OrderIntent(
            symbol="BTC_USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            order_type=OrderType.MARKET,
            timestamp=now,
        )

        order = order_intent_to_request(intent, confidence=0.85)

        assert order.confidence == 0.85


# ============================================================================
# Tests for SignalProcessor
# ============================================================================


class TestSignalProcessor:
    """Tests for SignalProcessor class."""

    def test_process_single_long_signal(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should process a single long signal into an order."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer)
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long")

        order = processor.process_signal(signal, empty_portfolio, current_price=50000)

        assert order is not None
        assert isinstance(order, OrderRequest)
        assert order.symbol == "BTC_USDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("0.5")

    def test_process_flat_signal_no_position(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should return None for flat signal with no position."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer)
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="flat")

        order = processor.process_signal(signal, empty_portfolio, current_price=50000)

        assert order is None

    def test_process_flat_signal_with_position(
        self, now: datetime, portfolio_with_long: PortfolioState
    ) -> None:
        """Should close position for flat signal."""
        sizer = FixedQuantitySizer(default_quantity="0.1")  # Ignored for flat
        processor = SignalProcessor(sizer=sizer)
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="flat")

        order = processor.process_signal(signal, portfolio_with_long, current_price=50000)

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == Decimal("1.0")  # Full position

    def test_processor_with_strategy_id(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should include strategy_id in generated orders."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer, strategy_id="test_strategy")
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long")

        order = processor.process_signal(signal, empty_portfolio, current_price=50000)

        assert order is not None
        assert order.strategy_id == "test_strategy"

    def test_processor_with_default_confidence(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should use default confidence when not in signal."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer, default_confidence=0.9)
        signal = Signal(symbol="BTC_USDT", timestamp=now, direction="long")

        order = processor.process_signal(signal, empty_portfolio, current_price=50000)

        assert order is not None
        assert order.confidence == 0.9

    def test_processor_confidence_from_signal_metadata(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should prefer confidence from signal metadata."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer, default_confidence=0.5)
        signal = Signal(
            symbol="BTC_USDT",
            timestamp=now,
            direction="long",
            metadata={"confidence": 0.95},
        )

        order = processor.process_signal(signal, empty_portfolio, current_price=50000)

        assert order is not None
        assert order.confidence == 0.95


class TestSignalProcessorBatch:
    """Tests for batch signal processing."""

    def test_process_multiple_signals(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should process multiple signals."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer)
        signals = [
            Signal(symbol="BTC_USDT", timestamp=now, direction="long"),
            Signal(symbol="ETH_USDT", timestamp=now, direction="short"),
        ]
        prices = {"BTC_USDT": 50000, "ETH_USDT": 3000}

        result = processor.process_signals(signals, empty_portfolio, prices)

        assert isinstance(result, ProcessingResult)
        assert len(result.orders) == 2
        assert result.orders[0].symbol == "BTC_USDT"
        assert result.orders[1].symbol == "ETH_USDT"

    def test_batch_skips_flat_no_position(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should skip flat signals with no position."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer)
        signals = [
            Signal(symbol="BTC_USDT", timestamp=now, direction="long"),
            Signal(symbol="ETH_USDT", timestamp=now, direction="flat"),  # No position
        ]
        prices = {"BTC_USDT": 50000, "ETH_USDT": 3000}

        result = processor.process_signals(signals, empty_portfolio, prices)

        assert len(result.orders) == 1
        assert len(result.skipped) == 1
        assert result.skipped[0].symbol == "ETH_USDT"

    def test_batch_handles_missing_price(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should record error for missing price."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer)
        signals = [
            Signal(symbol="BTC_USDT", timestamp=now, direction="long"),
            Signal(symbol="XYZ_USDT", timestamp=now, direction="long"),  # No price
        ]
        prices = {"BTC_USDT": 50000}  # Missing XYZ_USDT

        result = processor.process_signals(signals, empty_portfolio, prices)

        assert len(result.orders) == 1
        assert len(result.errors) == 1
        assert result.errors[0][0].symbol == "XYZ_USDT"
        assert "No price available" in result.errors[0][1]

    def test_empty_signals_list(
        self, now: datetime, empty_portfolio: PortfolioState
    ) -> None:
        """Should handle empty signals list."""
        sizer = FixedQuantitySizer(default_quantity="0.5")
        processor = SignalProcessor(sizer=sizer)

        result = processor.process_signals([], empty_portfolio, {})

        assert len(result.orders) == 0
        assert len(result.skipped) == 0
        assert len(result.errors) == 0


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_default_values(self) -> None:
        """Should have empty lists by default."""
        result = ProcessingResult()

        assert result.orders == []
        assert result.skipped == []
        assert result.errors == []
