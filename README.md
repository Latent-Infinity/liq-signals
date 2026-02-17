# liq-signals

Within the Latent Infinity Quant (LIQ) ecosystem, `liq-signals` defines the standard interface for signal providers and provides utilities for converting signals into executable orders.

## Installation

```bash
uv pip install liq-signals
```

## Quick Start

```python
from datetime import datetime, timezone
from decimal import Decimal

from liq.core import PortfolioState
from liq.signals import (
    Signal,
    SignalProcessor,
    TargetWeightSizer,
    FixedQuantitySizer,
)

# Create a signal
signal = Signal(
    symbol="BTC_USDT",
    timestamp=datetime.now(timezone.utc),
    direction="long",
    strength=0.8,
    target_weight=0.1,  # Target 10% portfolio allocation
)

# Set up portfolio state
portfolio = PortfolioState(
    cash=Decimal("100000"),
    equity=Decimal("100000"),
    positions={},
    timestamp=datetime.now(timezone.utc),
)

# Create processor with target weight sizing
processor = SignalProcessor(
    sizer=TargetWeightSizer(max_weight=0.2),
    strategy_id="momentum_v1",
)

# Convert signal to order
order = processor.process_signal(signal, portfolio, current_price=50000)
# order is an OrderRequest ready for liq-sim or broker submission
```

## Signal Data Structure

The `Signal` dataclass represents a trading signal (what to trade, not how much):

```python
from liq.signals import Signal

signal = Signal(
    symbol="BTC_USDT",           # Instrument symbol
    timestamp=datetime.now(tz),  # When signal was generated
    direction="long",            # "long", "short", or "flat"
    strength=0.8,                # Signal strength [0, 1], default 1.0
    target_weight=0.1,           # Target portfolio weight (optional)
    horizon=60,                  # Expected holding period in minutes (optional)
    metadata={"model": "lstm"},  # Custom metadata (optional)
)

# Normalize timestamp to UTC
utc_ts = signal.normalized_timestamp()
```

## Signal Providers

Implement the `SignalProvider` protocol to integrate any model or strategy:

```python
from liq.signals import SignalProvider, BaseSignalProvider, Signal

class MyStrategy(BaseSignalProvider):
    def __init__(self):
        super().__init__(
            name="my_strategy",
            symbols=["BTC_USDT", "ETH_USDT"],
            lookback=100,  # Required history bars
        )

    def generate_signals(self, data=None, portfolio_state=None):
        # Your signal generation logic here
        for symbol in self.symbols:
            yield Signal(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                direction="long",
                strength=0.75,
            )
```

### FileSignalProvider

Load signals from CSV, JSON, or JSONL files for backtesting:

```python
from pathlib import Path
from liq.signals import FileSignalProvider

# From JSON
provider = FileSignalProvider(Path("signals.json"))
signals = list(provider.generate_signals())

# Filter by symbols
provider = FileSignalProvider(
    Path("signals.csv"),
    symbols=["BTC_USDT"],  # Only load these symbols
)
```

Supported formats:

**CSV:**
```csv
symbol,timestamp,direction,strength,target_weight
BTC_USDT,2024-01-15T10:30:00Z,long,0.8,0.1
ETH_USDT,2024-01-15T10:30:00Z,short,0.6,0.05
```

**JSON:**
```json
[
  {"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"}
]
```

**JSONL:**
```
{"symbol": "BTC_USDT", "timestamp": "2024-01-15T10:30:00Z", "direction": "long"}
{"symbol": "ETH_USDT", "timestamp": "2024-01-15T10:35:00Z", "direction": "short"}
```

## Signal Sizing

Convert signals to sized order intents using sizing strategies.

### FixedQuantitySizer

Uses a fixed quantity for all signals:

```python
from liq.signals import FixedQuantitySizer, signal_to_order_intent

sizer = FixedQuantitySizer(default_quantity="0.5")
intent = sizer.size(signal, portfolio_state, current_price=50000)
# intent.quantity = 0.5 BTC
```

### TargetWeightSizer

Computes quantity from target portfolio weight:

```python
from liq.signals import TargetWeightSizer

sizer = TargetWeightSizer(
    max_weight=0.2,        # Cap at 20% per position
    min_order_value="100", # Filter orders < $100
)

# Signal with target_weight=0.1 and $100k equity at $50k/BTC
# Target value = $10k, quantity = 0.2 BTC
intent = sizer.size(signal, portfolio_state, current_price=50000)
```

Target weight sizing automatically:
- Uses `signal.target_weight` when available
- Falls back to `signal.strength * max_weight` otherwise
- Computes position delta (current vs target)
- Handles position increases, decreases, and closures

## Signal Processing Pipeline

`SignalProcessor` orchestrates the full signal-to-order pipeline:

```python
from liq.signals import SignalProcessor, TargetWeightSizer

processor = SignalProcessor(
    sizer=TargetWeightSizer(max_weight=0.15),
    strategy_id="mean_reversion",
    default_confidence=0.8,
)

# Process single signal
order = processor.process_signal(signal, portfolio, current_price=50000)

# Process batch
result = processor.process_signals(
    signals=[sig1, sig2, sig3],
    portfolio_state=portfolio,
    prices={"BTC_USDT": 50000, "ETH_USDT": 3000},
)

print(f"Orders: {len(result.orders)}")
print(f"Skipped: {len(result.skipped)}")  # e.g., flat with no position
print(f"Errors: {len(result.errors)}")    # e.g., missing price
```

## Integration with liq-sim

The output `OrderRequest` is ready for simulation:

```python
from liq.sim import Simulator
from liq.sim.config import ProviderConfig, SimulatorConfig
from liq.signals import SignalProcessor, TargetWeightSizer

# Generate orders from signals
processor = SignalProcessor(sizer=TargetWeightSizer())
result = processor.process_signals(signals, portfolio, prices)

# Run simulation
sim = Simulator(
    provider_config=ProviderConfig(name="binance"),
    config=SimulatorConfig(initial_capital=Decimal("100000")),
)
sim_result = sim.run(orders=result.orders, bars=bars)
```

## Exception Handling

```python
from liq.signals import (
    SizingError,
    InsufficientDataError,
    RiskConstraintError,
    InvalidSignalError,
)

try:
    order = processor.process_signal(signal, portfolio, current_price)
except InsufficientDataError as e:
    # Missing price or portfolio data
    print(f"Missing data: {e.message}")
except RiskConstraintError as e:
    # Position would exceed limits
    print(f"Risk limit: {e.constraint_name} = {e.constraint_value}")
except InvalidSignalError as e:
    # Malformed signal
    print(f"Invalid signal: {e.message}")
except SizingError as e:
    # Generic sizing failure
    print(f"Sizing failed: {e}")
```

## API Reference

### Core Types

```python
from liq.signals import (
    # Signal data
    Signal,
    Direction,  # Literal["long", "short", "flat"]

    # Provider protocol
    SignalProvider,
    BaseSignalProvider,
    FileSignalProvider,
)
```

### Sizing

```python
from liq.signals import (
    # Protocol and types
    SignalSizer,
    OrderIntent,

    # Implementations
    FixedQuantitySizer,
    TargetWeightSizer,

    # Utilities
    signal_to_order_intent,
    direction_to_side,
)
```

### Processing

```python
from liq.signals import (
    SignalProcessor,
    ProcessingResult,
    order_intent_to_request,
)
```

### Exceptions

```python
from liq.signals import (
    SizingError,           # Base exception
    InsufficientDataError, # Missing data
    RiskConstraintError,   # Risk limit violated
    InvalidSignalError,    # Malformed signal
)
```
