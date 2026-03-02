"""Tests for temporal correctness in signal handling.

Following TDD: Tests verify timestamp handling and UTC normalization.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from liq.signals import Signal


class TestSignalTimestampNormalization:
    """Tests for Signal.normalized_timestamp() method."""

    def test_naive_datetime_becomes_utc(self) -> None:
        """Naive datetime should be treated as UTC."""
        naive_ts = datetime(2024, 1, 15, 10, 30, 0)
        sig = Signal(symbol="BTC_USDT", timestamp=naive_ts, direction="long")

        normalized = sig.normalized_timestamp()

        assert normalized.tzinfo == UTC
        assert normalized.hour == 10  # Same hour, just tagged as UTC

    def test_utc_datetime_preserved(self) -> None:
        """UTC datetime should be preserved."""
        utc_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        sig = Signal(symbol="BTC_USDT", timestamp=utc_ts, direction="long")

        normalized = sig.normalized_timestamp()

        assert normalized == utc_ts
        assert normalized.tzinfo == UTC

    def test_other_timezone_converted_to_utc(self) -> None:
        """Non-UTC timezone should be converted to UTC."""
        # EST is UTC-5
        est = ZoneInfo("America/New_York")
        est_ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=est)
        sig = Signal(symbol="BTC_USDT", timestamp=est_ts, direction="long")

        normalized = sig.normalized_timestamp()

        assert normalized.tzinfo == UTC
        # 10:30 EST = 15:30 UTC (EST is UTC-5)
        assert normalized.hour == 15

    def test_utc_offset_timezone_converted(self) -> None:
        """Fixed offset timezone should be converted to UTC."""
        offset_tz = timezone(timedelta(hours=3))  # UTC+3
        ts_with_offset = datetime(2024, 1, 15, 10, 30, 0, tzinfo=offset_tz)
        sig = Signal(symbol="BTC_USDT", timestamp=ts_with_offset, direction="long")

        normalized = sig.normalized_timestamp()

        assert normalized.tzinfo == UTC
        # 10:30 UTC+3 = 07:30 UTC
        assert normalized.hour == 7


class TestSignalDataIntegrity:
    """Tests for Signal data structure integrity."""

    def test_signal_is_frozen(self) -> None:
        """Signal should be immutable (frozen dataclass)."""
        sig = Signal(
            symbol="BTC_USDT",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            direction="long",
        )

        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            sig.symbol = "ETH_USDT"  # type: ignore[misc]

    def test_signal_defaults(self) -> None:
        """Signal should have sensible defaults."""
        sig = Signal(
            symbol="BTC_USDT",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            direction="long",
        )

        assert sig.strength == 1.0
        assert sig.target_weight is None
        assert sig.horizon is None
        assert sig.metadata == {}

    def test_signal_with_all_fields(self) -> None:
        """Signal with all fields should preserve values."""
        metadata = {"strategy": "momentum", "confidence": 0.95}
        sig = Signal(
            symbol="BTC_USDT",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            direction="short",
            strength=0.8,
            target_weight=0.1,
            horizon=60,
            metadata=metadata,
        )

        assert sig.symbol == "BTC_USDT"
        assert sig.direction == "short"
        assert sig.strength == 0.8
        assert sig.target_weight == 0.1
        assert sig.horizon == 60
        assert sig.metadata == metadata

    def test_direction_type(self) -> None:
        """Direction should accept valid values."""
        for direction in ["long", "short", "flat"]:
            sig = Signal(
                symbol="BTC_USDT",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                direction=direction,  # type: ignore[arg-type]
            )
            assert sig.direction == direction


class TestSignalMetadataIntegrity:
    """Tests for metadata handling in signals."""

    def test_metadata_default_is_isolated(self) -> None:
        """Each signal should have independent metadata dict."""
        sig1 = Signal(
            symbol="BTC_USDT",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            direction="long",
        )
        sig2 = Signal(
            symbol="ETH_USDT",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            direction="short",
        )

        # Modifying metadata dict (if we could) should not affect other signals
        # Since Signal is frozen, we can't modify, but verify they're separate
        assert sig1.metadata is not sig2.metadata

    def test_metadata_preserved_through_roundtrip(self) -> None:
        """Metadata should survive signal creation."""
        original_metadata = {
            "model_version": "v1.2.3",
            "features_used": ["rsi", "macd", "volume"],
            "prediction_score": 0.87,
        }

        sig = Signal(
            symbol="BTC_USDT",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            direction="long",
            metadata=original_metadata,
        )

        assert sig.metadata == original_metadata
        assert sig.metadata["model_version"] == "v1.2.3"
        assert sig.metadata["features_used"] == ["rsi", "macd", "volume"]


class TestSignalTimestampEdgeCases:
    """Tests for timestamp edge cases."""

    def test_midnight_utc(self) -> None:
        """Midnight UTC should normalize correctly."""
        midnight = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        sig = Signal(symbol="BTC_USDT", timestamp=midnight, direction="flat")

        normalized = sig.normalized_timestamp()

        assert normalized.hour == 0
        assert normalized.minute == 0
        assert normalized.second == 0

    def test_end_of_day_utc(self) -> None:
        """End of day UTC should normalize correctly."""
        eod = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)
        sig = Signal(symbol="BTC_USDT", timestamp=eod, direction="flat")

        normalized = sig.normalized_timestamp()

        assert normalized.hour == 23
        assert normalized.minute == 59
        assert normalized.second == 59

    def test_dst_transition(self) -> None:
        """Signals during DST transition should normalize correctly."""
        # US Eastern switches to EDT (UTC-4) on March 10, 2024
        eastern = ZoneInfo("America/New_York")
        # 3:30 AM EDT (after DST transition)
        during_dst = datetime(2024, 3, 10, 3, 30, 0, tzinfo=eastern)
        sig = Signal(symbol="BTC_USDT", timestamp=during_dst, direction="long")

        normalized = sig.normalized_timestamp()

        assert normalized.tzinfo == UTC
        # 3:30 AM EDT = 7:30 AM UTC
        assert normalized.hour == 7
        assert normalized.minute == 30

    def test_microsecond_precision(self) -> None:
        """Microsecond precision should be preserved."""
        ts_with_micros = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)
        sig = Signal(symbol="BTC_USDT", timestamp=ts_with_micros, direction="long")

        normalized = sig.normalized_timestamp()

        assert normalized.microsecond == 123456
