import json
from datetime import datetime
from pathlib import Path

import pytest

from liq.signals import FileSignalProvider, Signal


def test_signal_timestamp_normalized() -> None:
    sig = Signal(symbol="BTC_USDT", timestamp=datetime(2024, 1, 1, 0, 0), direction="long")
    normalized = sig.normalized_timestamp()
    assert normalized.tzinfo is not None
    assert normalized.tzinfo.utcoffset(normalized) is not None


def test_file_signal_provider_json(tmp_path: Path) -> None:
    path = tmp_path / "signals.json"
    path.write_text(
        """[
        {"symbol": "BTC_USDT", "timestamp": "2024-01-01T00:00:00Z", "direction": "long", "strength": 0.9},
        {"symbol": "ETH_USDT", "timestamp": "2024-01-01T00:05:00+00:00", "direction": "short"}
    ]"""
    )
    provider = FileSignalProvider(path)
    signals = list(provider.generate_signals())
    assert len(signals) == 2
    assert signals[0].symbol == "BTC_USDT"
    assert signals[0].direction == "long"
    assert signals[0].normalized_timestamp().tzinfo is not None


def test_file_signal_provider_csv_filters_symbols(tmp_path: Path) -> None:
    path = tmp_path / "signals.csv"
    path.write_text(
        "symbol,timestamp,direction,strength\n"
        "BTC_USDT,2024-01-01T00:00:00Z,long,1.0\n"
        "ETH_USDT,2024-01-01T00:01:00Z,short,0.5\n"
    )
    provider = FileSignalProvider(path, symbols=["BTC_USDT"])
    signals = list(provider.generate_signals())
    assert len(signals) == 1
    assert signals[0].symbol == "BTC_USDT"


def test_file_signal_provider_jsonl_metadata(tmp_path: Path) -> None:
    path = tmp_path / "signals.jsonl"
    path.write_text(
        '{"symbol": "BTC_USDT", "timestamp": "2024-01-01T00:00:00Z", "direction": "long", "strength": 0.8, "note": "demo"}\n'
        '{"symbol": "BTC_USDT", "timestamp": "2024-01-01T00:05:00Z", "direction": "flat"}\n'
    )
    provider = FileSignalProvider(path)
    signals = list(provider.generate_signals())
    assert len(signals) == 2
    assert signals[0].metadata.get("note") == "demo"
    assert signals[1].direction == "flat"


def test_file_signal_provider_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "signals.txt"
    path.write_text("dummy")
    provider = FileSignalProvider(path)
    with pytest.raises(ValueError):
        list(provider.generate_signals())


def test_file_signal_provider_json_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "signals_wrapped.json"
    path.write_text(
        json.dumps(
            {
                "signals": [
                    {
                        "symbol": "ETH_USDT",
                        "timestamp": "2024-01-01T01:00:00+00:00",
                        "direction": "short",
                        "strength": 0.6,
                    }
                ]
            }
        )
    )
    provider = FileSignalProvider(path)
    signals = list(provider.generate_signals())
    assert len(signals) == 1
    assert signals[0].normalized_timestamp().tzinfo is not None
