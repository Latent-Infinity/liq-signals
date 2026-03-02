"""Tests for baseline SignalProvider implementations (Phase 0b Task 0b.8).

Two baselines:
    PassiveBaseline: equal-weight buy-and-hold
    NaiveActiveBaseline: periodic rebalance to equal weights
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from liq.core import Bar, OrderRequest, OrderSide, PortfolioState, Position

from liq.signals import BaseSignalProvider, Signal, SignalProcessor, TargetWeightSizer


def _has_liq_sim() -> bool:
    """Check if liq-sim is available in this environment."""
    try:
        import liq.sim  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# PassiveBaseline tests
# ---------------------------------------------------------------------------


class TestPassiveBaselineStructure:
    """PassiveBaseline satisfies the SignalProvider protocol."""

    def test_is_subclass_of_base(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        assert issubclass(PassiveBaseline, BaseSignalProvider)

    def test_name_property(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        assert provider.name == "passive_equal_weight"

    def test_symbols_property(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        assert provider.symbols == ["BTC-USD", "ETH-USD"]

    def test_required_history_is_zero(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD"])
        assert provider.required_history == 0

    def test_custom_name(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD"], name="my_passive")
        assert provider.name == "my_passive"


class TestPassiveBaselineSignals:
    """PassiveBaseline generates correct equal-weight long signals."""

    def test_generates_one_signal_per_symbol(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        assert len(signals) == 2

    def test_all_signals_are_long(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD", "ETH-USD", "SOL-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        assert all(s.direction == "long" for s in signals)

    def test_target_weights_are_equal(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        provider = PassiveBaseline(symbols=symbols)
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        expected_weight = 1.0 / len(symbols)
        for s in signals:
            assert s.target_weight == pytest.approx(expected_weight)

    def test_signals_have_correct_symbols(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        symbols = ["BTC-USD", "ETH-USD"]
        provider = PassiveBaseline(symbols=symbols)
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        signal_symbols = {s.symbol for s in signals}
        assert signal_symbols == set(symbols)

    def test_signals_use_provided_timestamp(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD"])
        ts = datetime(2025, 6, 15, 12, 30, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        assert signals[0].timestamp == ts

    def test_signals_have_full_strength(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        assert signals[0].strength == 1.0

    def test_single_symbol_gets_weight_one(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        assert signals[0].target_weight == pytest.approx(1.0)

    def test_default_timestamp_is_utc_now(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD"])
        before = datetime.now(UTC)
        signals = list(provider.generate_signals())
        after = datetime.now(UTC)
        assert before <= signals[0].timestamp <= after

    def test_returns_valid_signal_instances(self) -> None:
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        for s in signals:
            assert isinstance(s, Signal)


# ---------------------------------------------------------------------------
# NaiveActiveBaseline tests
# ---------------------------------------------------------------------------


class TestNaiveActiveBaselineStructure:
    """NaiveActiveBaseline satisfies the SignalProvider protocol."""

    def test_is_subclass_of_base(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        assert issubclass(NaiveActiveBaseline, BaseSignalProvider)

    def test_name_property(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        assert provider.name == "naive_active_rebalance"

    def test_symbols_property(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        assert provider.symbols == ["BTC-USD", "ETH-USD"]

    def test_required_history_is_zero(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD"])
        assert provider.required_history == 0


class TestNaiveActiveBaselineSignals:
    """NaiveActiveBaseline generates rebalance signals based on portfolio drift."""

    def _make_portfolio(
        self,
        positions: dict[str, tuple[Decimal, Decimal]],
        cash: Decimal = Decimal("0"),
    ) -> PortfolioState:
        """Create portfolio. positions: symbol → (quantity, current_price)."""
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        return PortfolioState(
            cash=cash,
            positions={
                sym: Position(
                    symbol=sym,
                    quantity=qty,
                    average_price=price,
                    realized_pnl=Decimal("0"),
                    current_price=price,
                    timestamp=ts,
                )
                for sym, (qty, price) in positions.items()
            },
            timestamp=ts,
        )

    def test_generates_signals_for_all_symbols(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        portfolio = self._make_portfolio(
            {
                "BTC-USD": (Decimal("1"), Decimal("50000")),
                "ETH-USD": (Decimal("10"), Decimal("3000")),
            },
        )
        signals = list(provider.generate_signals(data=ts, portfolio_state=portfolio))
        assert len(signals) == 2

    def test_all_signals_are_long(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        portfolio = self._make_portfolio(
            {
                "BTC-USD": (Decimal("1"), Decimal("50000")),
                "ETH-USD": (Decimal("10"), Decimal("3000")),
            },
        )
        signals = list(provider.generate_signals(data=ts, portfolio_state=portfolio))
        assert all(s.direction == "long" for s in signals)

    def test_target_weights_are_equal(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        symbols = ["BTC-USD", "ETH-USD"]
        provider = NaiveActiveBaseline(symbols=symbols)
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        portfolio = self._make_portfolio(
            {
                "BTC-USD": (Decimal("1"), Decimal("50000")),
                "ETH-USD": (Decimal("10"), Decimal("3000")),
            },
        )
        signals = list(provider.generate_signals(data=ts, portfolio_state=portfolio))
        expected_weight = 1.0 / len(symbols)
        for s in signals:
            assert s.target_weight == pytest.approx(expected_weight)

    def test_no_portfolio_state_still_generates_signals(self) -> None:
        """Without portfolio state, acts like passive baseline."""
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        assert len(signals) == 2

    def test_signals_have_rebalance_metadata(self) -> None:
        """Signals should carry baseline_type metadata."""
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))
        assert signals[0].metadata.get("baseline_type") == "naive_active_rebalance"

    def test_custom_rebalance_threshold(self) -> None:
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(
            symbols=["BTC-USD", "ETH-USD"],
            rebalance_threshold=0.10,
        )
        assert provider._rebalance_threshold == 0.10  # noqa: SLF001


# ---------------------------------------------------------------------------
# End-to-end: signal → order pipeline evaluability
# ---------------------------------------------------------------------------


class TestBaselineEvaluability:
    """Baselines produce signals that can be processed into OrderRequests."""

    def _make_empty_portfolio(self) -> PortfolioState:
        return PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

    def test_passive_signals_produce_orders(self) -> None:
        """PassiveBaseline signals → SignalProcessor → OrderRequests."""
        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))

        processor = SignalProcessor(sizer=TargetWeightSizer(max_weight=1.0))
        portfolio = self._make_empty_portfolio()
        prices = {"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")}

        result = processor.process_signals(signals, portfolio, prices)
        assert len(result.orders) == 2
        assert len(result.errors) == 0
        for order in result.orders:
            assert isinstance(order, OrderRequest)
            assert order.side == OrderSide.BUY
            assert order.quantity > 0

    def test_naive_active_signals_produce_orders(self) -> None:
        """NaiveActiveBaseline signals → SignalProcessor → OrderRequests."""
        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD", "ETH-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))

        processor = SignalProcessor(sizer=TargetWeightSizer(max_weight=1.0))
        portfolio = self._make_empty_portfolio()
        prices = {"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")}

        result = processor.process_signals(signals, portfolio, prices)
        assert len(result.orders) == 2
        assert len(result.errors) == 0

    @pytest.mark.skipif(
        not _has_liq_sim(), reason="liq-sim not installed in this environment"
    )
    def test_passive_evaluable_through_simulator(self) -> None:
        """Full pipeline: PassiveBaseline → SignalProcessor → Simulator."""
        from liq.sim.config import ProviderConfig, SimulatorConfig
        from liq.sim.simulator import Simulator

        from liq.signals.baselines import PassiveBaseline

        provider = PassiveBaseline(symbols=["BTC-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))

        processor = SignalProcessor(sizer=TargetWeightSizer(max_weight=1.0))
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=ts,
        )
        prices = {"BTC-USD": Decimal("50000")}
        result = processor.process_signals(signals, portfolio, prices)
        assert len(result.orders) >= 1

        # Create minimal bars for the simulator
        bars = [
            Bar(
                timestamp=ts + timedelta(minutes=i * 5),
                symbol="BTC-USD",
                open=Decimal("50000"),
                high=Decimal("50500"),
                low=Decimal("49500"),
                close=Decimal("50100"),
                volume=Decimal("100"),
            )
            for i in range(5)
        ]

        sim = Simulator(
            provider_config=ProviderConfig(
                name="test_crypto",
                asset_classes=["crypto"],
                fee_model="ZeroCommission",
                slippage_model="VolumeWeighted",
                slippage_params={"base_bps": "0", "volume_impact": "0"},
            ),
            config=SimulatorConfig(initial_capital=Decimal("100000")),
        )
        sim_result = sim.run(orders=result.orders, bars=bars)
        # Simulation completes without error
        assert len(sim_result.equity_curve) > 0
        assert sim_result.equity_curve[-1][1] > 0

    @pytest.mark.skipif(
        not _has_liq_sim(), reason="liq-sim not installed in this environment"
    )
    def test_naive_active_evaluable_through_simulator(self) -> None:
        """Full pipeline: NaiveActiveBaseline → SignalProcessor → Simulator."""
        from liq.sim.config import ProviderConfig, SimulatorConfig
        from liq.sim.simulator import Simulator

        from liq.signals.baselines import NaiveActiveBaseline

        provider = NaiveActiveBaseline(symbols=["BTC-USD"])
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        signals = list(provider.generate_signals(data=ts))

        processor = SignalProcessor(sizer=TargetWeightSizer(max_weight=1.0))
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=ts,
        )
        prices = {"BTC-USD": Decimal("50000")}
        result = processor.process_signals(signals, portfolio, prices)
        assert len(result.orders) >= 1

        bars = [
            Bar(
                timestamp=ts + timedelta(minutes=i * 5),
                symbol="BTC-USD",
                open=Decimal("50000"),
                high=Decimal("50500"),
                low=Decimal("49500"),
                close=Decimal("50100"),
                volume=Decimal("100"),
            )
            for i in range(5)
        ]

        sim = Simulator(
            provider_config=ProviderConfig(
                name="test_crypto",
                asset_classes=["crypto"],
                fee_model="ZeroCommission",
                slippage_model="VolumeWeighted",
                slippage_params={"base_bps": "0", "volume_impact": "0"},
            ),
            config=SimulatorConfig(initial_capital=Decimal("100000")),
        )
        sim_result = sim.run(orders=result.orders, bars=bars)
        assert len(sim_result.equity_curve) > 0
        assert sim_result.equity_curve[-1][1] > 0
