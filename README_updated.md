
# Quant Research Framework

A modular framework for developing, backtesting, statistically evaluating, and paper-trading **crypto market-making and mid-frequency trading (MFT)** strategies before risking real capital.

The framework is designed so that **strategies, alpha signals, and market data connectors are interchangeable**. While the current implementation focuses on crypto markets, the same architecture can be adapted to other markets (e.g. Indian equities/futures) by replacing the data connector.

---

## Architecture

```text
Market Data
     │
     ▼
 Signal Engine
     │
     ▼
 Strategy Engine
     │
     ▼
 Risk Management
     │
     ▼
 Execution Simulator
     │
     ▼
 Performance Analytics
     │
     ▼
 Paper Trading Dashboard
```

---

## Current Implementations

### Alpha Models
- Avellaneda-Stoikov inventory-skew market making (baseline)
- Bar Portion mean-reversion alpha (Cornell FE Manhattan, 2024)

### Framework Features

- Modular strategy interface
- Historical data downloader
- Backtesting engine
- Paper trading dashboard
- Transaction costs & slippage
- Dynamic position sizing
- Inventory limits
- Drawdown circuit breaker
- Statistical significance testing
- Bootstrap strategy comparison

---

## Components

### 1. Live Paper Trading Dashboard

- Live websocket market data
- Real-time quotes
- Simulated fills
- Inventory tracking
- Live PnL
- Order book visualization
- Risk monitoring dashboard

```bash
streamlit run pmm_app.py
```

### 2. Research & Backtesting Engine

- Historical data download
- Plug-and-play strategies
- Cost-adjusted backtests
- Baseline vs alpha comparison
- Regression analysis
- Bootstrap significance testing
- Performance metrics

```bash
python backtest_framework.py
```

---

## Plugging in a New Strategy

```python
class MyStrategy:
    name = "My Strategy"

    def get_quotes(self, mid, inventory, sigma, signal):
        return bid, ask, spread
```

No changes to the backtest engine are required.

---

## Project Roadmap

Completed

- ✅ Backtesting engine
- ✅ Paper trading
- ✅ Risk management
- ✅ Statistical evaluation
- ✅ Modular strategy interface

Planned

- ⬜ Order Flow Imbalance
- ⬜ Queue imbalance
- ⬜ Microprice models
- ⬜ Cross-exchange arbitrage
- ⬜ Walk-forward optimization
- ⬜ Portfolio backtesting
- ⬜ Indian market data connector
- ⬜ Level-2 order book replay

---

## Current Limitations

Current focus is on providing a robust **research framework** rather than production execution.

Future improvements include:

- Queue position modelling
- Partial fills
- Latency simulation
- Adverse selection modelling
- Production execution support

---

## References

- Avellaneda & Stoikov (2008), *High Frequency Trading in a Limit Order Book*
- Stoikov et al. (2024), *Market Making in Crypto*

---

## Tech Stack

Python • Streamlit • ccxt • ccxt.pro • pandas • numpy • scipy
