# PMM-BP: Crypto Market Making Strategy Simulator

A paper-trading framework for testing crypto market-making strategies — live and backtested — before risking real capital. Built around a live Streamlit dashboard plus a standalone backtesting/statistics module, so new signals can be plugged in and rigorously evaluated before going live.

> Implements the **Avellaneda-Stoikov** inventory-skew quoting model ([AS08]) as a baseline, and the **Bar Portion** mean-reversion alpha signal from *Market Making in Crypto* (Stoikov, Zhuang, Chen, Zhang, Wang, Li, Shan — Cornell Financial Engineering Manhattan, Dec 2024) as a candidate alpha overlay.

---

## Why this exists

Most market-making code either trades real money immediately or backtests on a single asset with no cost model and calls it a day. This project is meant to be the missing middle step: a sandbox where a strategy can be paper-traded against **live** exchange data, and separately stress-tested against **historical** data with realistic transaction costs, slippage, and statistical significance testing — before anyone deploys it with real funds.

The goal is for other people (juniors, quant club members, anyone exploring algo trading) to be able to drop in their own signal and get an honest answer to: *does this actually add edge, or am I just seeing noise?*

---

## Components

### 1. Live paper-trading dashboard (`pmm_app.py`)
A Streamlit app that:
- Streams live order book + ticker data from Binance/Kraken via `ccxt.pro` websockets
- Computes the Bar Portion signal from rolling 1-minute candles
- Quotes bid/ask using an inventory-skew + signal-skew reservation price
- Simulates fills against live market data (demo mode — no real orders are placed)
- Tracks running inventory and PnL in real time on a live dashboard

Run with:
```bash
streamlit run pmm_app.py
```

### 2. Backtesting & statistics framework (`backtest_framework.py`)
A standalone module for rigorous offline evaluation:
- **Historical data download** via `ccxt.fetch_ohlcv` (free, no API key required)
- **Transaction cost model** — configurable maker/taker fees + slippage applied to every simulated fill
- **Baseline vs. signal comparison** — a no-signal pure inventory-skew market maker (`BaselineInventoryMM`) vs. the Bar-Portion-augmented strategy (`BarPortionMM`), run through an identical cost-adjusted backtest engine so any performance gap is attributable to the signal itself
- **Signal significance testing** — regression of forward returns on the signal value (p-value, R², quintile monotonicity), replicating the methodology used in the source paper
- **Strategy significance testing** — paired t-test + bootstrap confidence interval on the return difference between two strategies, to check whether an apparent edge is statistically real or just a lucky equity curve

Run with:
```bash
python backtest_framework.py
```

---

## Plugging in your own strategy

Every strategy implements one interface:

```python
class MyStrategy:
    name = "My Strategy"

    def get_quotes(self, mid, inventory, sigma, signal):
        # return (bid_price, ask_price, spread)
        ...
```

Drop your class into `compare_strategies()` alongside `BaselineInventoryMM` to get cost-adjusted PnL, Sharpe, drawdown, and significance testing against the baseline automatically — no need to rebuild the backtest engine.

---

## Known limitations (read before trusting any PnL number)

- Fill simulation assumes the order is fully filled the instant price crosses it — no partial fills, no queue position, no latency modeling.
- No adverse-selection modeling (market makers are statistically more likely to get filled right before the market moves against them — this isn't captured).
- Statistical significance testing assumes returns are independent across periods, which is a simplification for 1-minute crypto data.
- This is a research/educational sandbox, not a production trading system. The "Demo Mode" toggle in the live app does not currently route to real order placement even when switched off — live execution is intentionally unimplemented to keep this safe for experimentation.

---

## References

- Avellaneda, M., & Stoikov, S. (2008). *High frequency trading in a limit order book.* Quantitative Finance, 8, 217–224.
- Stoikov, S., Zhuang, E., Chen, H., Zhang, Q., Wang, S., Li, S., & Shan, C. (2024). *Market Making in Crypto.* Cornell Financial Engineering Manhattan.

---

## Stack

`Python` · `Streamlit` · `ccxt` / `ccxt.pro` · `pandas` / `numpy` · `scipy.stats`
