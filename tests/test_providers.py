"""Tests for signal providers.

Following TDD: Tests verify provider functionality and edge cases.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from liq.signals import (
    BaseSignalProvider,
    FileSignalProvider,
    Signal,
)


class TestBaseSignalProvider:
    """Tests for BaseSignalProvider class."""

    def test_base_provider_properties(self) -> None:
        """BaseSignalProvider should expose required properties."""
        provider = BaseSignalProvider(
            name="test_provider",
            symbols=["BTC_USDT", "ETH_USDT"],
            lookback=100,
        )

        assert provider.name == "test_provider"
        assert provider.symbols == ["BTC_USDT", "ETH_USDT"]
        assert provider.required_history == 100

    def test_base_provider_default_lookback(self) -> None:
        """Default lookback should be 0."""
        provider = BaseSignalProvider(
            name="test",
            symbols=["BTC_USDT"],
        )

        assert provider.required_history == 0

    def test_base_provider_symbols_is_copy(self) -> None:
        """Symbols list should be a copy to prevent mutation."""
        original_symbols = ["BTC_USDT", "ETH_USDT"]
        provider = BaseSignalProvider(
            name="test",
            symbols=original_symbols,
        )

        # Modifying original should not affect provider
        original_symbols.append("SOL_USDT")
        assert "SOL_USDT" not in provider.symbols


class TestFileSignalProviderCSV:
    """Tests for FileSignalProvider with CSV files."""

    def test_csv_basic_load(self, tmp_path: Path) -> None:
        """Should load basic CSV signals."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "symbol,timestamp,direction,strength\n"
            "BTC_USDT,2024-01-15T10:30:00Z,long,0.8\n"
            "ETH_USDT,2024-01-15T10:35:00+00:00,short,0.6\n"
        )

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 2
        assert signals[0].symbol == "BTC_USDT"
        assert signals[0].direction == "long"
        assert signals[0].strength == 0.8
        assert signals[1].symbol == "ETH_USDT"
        assert signals[1].direction == "short"

    def test_csv_with_ts_column_alias(self, tmp_path: Path) -> None:
        """Should accept 'ts' as timestamp column alias."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text("symbol,ts,direction\nBTC_USDT,2024-01-15T10:30:00Z,long\n")

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 1
        assert signals[0].timestamp.tzinfo is not None

    def test_csv_with_target_weight_and_horizon(self, tmp_path: Path) -> None:
        """Should parse target_weight and horizon fields."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "symbol,timestamp,direction,strength,target_weight,horizon\n"
            "BTC_USDT,2024-01-15T10:30:00Z,long,0.8,0.1,60\n"
        )

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert signals[0].target_weight == 0.1
        assert signals[0].horizon == 60

    def test_csv_with_empty_optional_fields(self, tmp_path: Path) -> None:
        """Empty target_weight and horizon should be None."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "symbol,timestamp,direction,strength,target_weight,horizon\n"
            "BTC_USDT,2024-01-15T10:30:00Z,long,0.8,,\n"
        )

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert signals[0].target_weight is None
        assert signals[0].horizon is None

    def test_csv_extra_columns_become_metadata(self, tmp_path: Path) -> None:
        """Extra columns should be captured in metadata."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "symbol,timestamp,direction,model_version,confidence\n"
            "BTC_USDT,2024-01-15T10:30:00Z,long,v1.0,0.95\n"
        )

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert signals[0].metadata["model_version"] == "v1.0"
        assert signals[0].metadata["confidence"] == "0.95"  # CSV values are strings

    def test_csv_skips_invalid_timestamps(self, tmp_path: Path) -> None:
        """Should skip rows with invalid timestamps."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "symbol,timestamp,direction\n"
            "BTC_USDT,2024-01-15T10:30:00Z,long\n"
            "ETH_USDT,not-a-timestamp,short\n"
            "SOL_USDT,2024-01-15T10:35:00Z,flat\n"
        )

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 2
        assert signals[0].symbol == "BTC_USDT"
        assert signals[1].symbol == "SOL_USDT"

    def test_csv_skips_empty_timestamp(self, tmp_path: Path) -> None:
        """Should skip rows with empty timestamps."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "symbol,timestamp,direction\n"
            "BTC_USDT,2024-01-15T10:30:00Z,long\n"
            "ETH_USDT,,short\n"
        )

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 1

    def test_csv_symbol_uppercased(self, tmp_path: Path) -> None:
        """Symbol should be uppercased."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            "symbol,timestamp,direction\nbtc_usdt,2024-01-15T10:30:00Z,long\n"
        )

        provider = FileSignalProvider(csv_path)
        signals = list(provider.generate_signals())

        assert signals[0].symbol == "BTC_USDT"


class TestFileSignalProviderJSON:
    """Tests for FileSignalProvider with JSON files."""

    def test_json_array_format(self, tmp_path: Path) -> None:
        """Should load JSON array format."""
        json_path = tmp_path / "signals.json"
        json_path.write_text(
            '[{"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"}]'
        )

        provider = FileSignalProvider(json_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 1
        assert signals[0].symbol == "BTC_USDT"

    def test_json_wrapped_format(self, tmp_path: Path) -> None:
        """Should load wrapped {'signals': [...]} format."""
        json_path = tmp_path / "signals.json"
        json_path.write_text(
            '{"signals": [{"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"}]}'
        )

        provider = FileSignalProvider(json_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 1

    def test_json_with_all_fields(self, tmp_path: Path) -> None:
        """Should parse all signal fields from JSON."""
        json_path = tmp_path / "signals.json"
        json_path.write_text(
            """[{
                "symbol": "BTC_USDT",
                "timestamp": "2024-01-15T10:30:00Z",
                "direction": "long",
                "strength": 0.9,
                "target_weight": 0.15,
                "horizon": 120,
                "extra_field": "custom_data"
            }]"""
        )

        provider = FileSignalProvider(json_path)
        signals = list(provider.generate_signals())

        assert signals[0].strength == 0.9
        assert signals[0].target_weight == 0.15
        assert signals[0].horizon == 120
        assert signals[0].metadata["extra_field"] == "custom_data"

    def test_json_empty_file(self, tmp_path: Path) -> None:
        """Empty JSON file should return empty list."""
        json_path = tmp_path / "signals.json"
        json_path.write_text("")

        provider = FileSignalProvider(json_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 0


class TestFileSignalProviderJSONL:
    """Tests for FileSignalProvider with JSONL files."""

    def test_jsonl_format(self, tmp_path: Path) -> None:
        """Should load JSONL format (one JSON per line)."""
        jsonl_path = tmp_path / "signals.jsonl"
        jsonl_path.write_text(
            '{"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"}\n'
            '{"symbol": "ETH_USDT", "timestamp": "2024-01-15T10:35:00Z", "direction": "short"}\n'
        )

        provider = FileSignalProvider(jsonl_path)
        signals = list(provider.generate_signals())

        assert len(signals) == 2
        assert signals[0].symbol == "BTC_USDT"
        assert signals[1].symbol == "ETH_USDT"


class TestFileSignalProviderFiltering:
    """Tests for symbol filtering in FileSignalProvider."""

    def test_filter_by_symbols(self, tmp_path: Path) -> None:
        """Should filter signals by symbol list."""
        json_path = tmp_path / "signals.json"
        json_path.write_text(
            """[
                {"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"},
                {"symbol": "ETH_USDT", "timestamp": "2024-01-15T10:35:00Z", "direction": "short"},
                {"symbol": "SOL_USDT", "timestamp": "2024-01-15T10:40:00Z", "direction": "flat"}
            ]"""
        )

        provider = FileSignalProvider(json_path, symbols=["BTC_USDT", "SOL_USDT"])
        signals = list(provider.generate_signals())

        assert len(signals) == 2
        symbols = [s.symbol for s in signals]
        assert "BTC_USDT" in symbols
        assert "SOL_USDT" in symbols
        assert "ETH_USDT" not in symbols

    def test_filter_empty_symbols_returns_all(self, tmp_path: Path) -> None:
        """Empty symbols list should return all signals."""
        json_path = tmp_path / "signals.json"
        json_path.write_text(
            """[
                {"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"},
                {"symbol": "ETH_USDT", "timestamp": "2024-01-15T10:35:00Z", "direction": "short"}
            ]"""
        )

        provider = FileSignalProvider(json_path, symbols=[])
        signals = list(provider.generate_signals())

        assert len(signals) == 2


class TestFileSignalProviderCaching:
    """Tests for signal caching behavior."""

    def test_signals_are_cached(self, tmp_path: Path) -> None:
        """Signals should be cached after first load."""
        json_path = tmp_path / "signals.json"
        json_path.write_text(
            '[{"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"}]'
        )

        provider = FileSignalProvider(json_path)

        # First call loads from file
        signals1 = list(provider.generate_signals())

        # Modify file (should not affect cached result)
        json_path.write_text(
            '[{"symbol": "ETH_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "short"}]'
        )

        # Second call should return cached signals
        signals2 = list(provider.generate_signals())

        assert signals1[0].symbol == "BTC_USDT"
        assert signals2[0].symbol == "BTC_USDT"  # Still cached


class TestFileSignalProviderProperties:
    """Tests for FileSignalProvider properties."""

    def test_provider_name(self, tmp_path: Path) -> None:
        """Provider name should be 'file_provider'."""
        json_path = tmp_path / "signals.json"
        json_path.write_text("[]")

        provider = FileSignalProvider(json_path)

        assert provider.name == "file_provider"

    def test_provider_required_history(self, tmp_path: Path) -> None:
        """File provider requires no history."""
        json_path = tmp_path / "signals.json"
        json_path.write_text("[]")

        provider = FileSignalProvider(json_path)

        assert provider.required_history == 0

    def test_provider_symbols_from_constructor(self, tmp_path: Path) -> None:
        """Symbols should match constructor argument."""
        json_path = tmp_path / "signals.json"
        json_path.write_text("[]")

        provider = FileSignalProvider(json_path, symbols=["BTC_USDT"])

        assert provider.symbols == ["BTC_USDT"]


class TestSignalProviderProtocol:
    """Tests verifying protocol compliance."""

    def test_file_signal_provider_implements_protocol(self, tmp_path: Path) -> None:
        """FileSignalProvider should implement SignalProvider protocol."""
        json_path = tmp_path / "signals.json"
        json_path.write_text("[]")

        provider = FileSignalProvider(json_path)

        # Verify protocol methods exist
        assert hasattr(provider, "generate_signals")
        assert hasattr(provider, "required_history")
        assert hasattr(provider, "symbols")
        assert hasattr(provider, "name")

        # Verify they're callable/accessible
        assert callable(provider.generate_signals)
        assert isinstance(provider.required_history, int)
        assert isinstance(provider.symbols, list)
        assert isinstance(provider.name, str)

    def test_base_signal_provider_can_be_subclassed(self) -> None:
        """BaseSignalProvider should be subclassable."""

        class CustomProvider(BaseSignalProvider):
            def generate_signals(
                self, data: Any | None = None, portfolio_state: Any | None = None
            ) -> Iterable[Signal]:
                return [
                    Signal(
                        symbol="TEST",
                        timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                        direction="long",
                    )
                ]

        provider = CustomProvider(name="custom", symbols=["TEST"])
        signals = list(provider.generate_signals())

        assert len(signals) == 1
        assert signals[0].symbol == "TEST"
