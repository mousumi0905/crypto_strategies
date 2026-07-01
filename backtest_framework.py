# ═══════════════════════════════════════════════════════════
# BACKTEST FRAMEWORK  v2.0
# ─────────────────────────────────────────────────────────
# Risk controls added (v2):
#   • Inventory cap   — hard limit on |position| size
#   • Circuit breaker — halts new orders past max drawdown
#   • Dynamic sizing  — order size scales with equity/price
#   • Transaction costs + slippage on every fill
#   • max_inventory_reached + pct_time_halted in metrics
#
# Usage:
#   python backtest_framework.py 30          # 30 days 1m
#   python backtest_framework.py 30 15m      # 30 days 15m
# ═══════════════════════════════════════════════════════════
import sys
import time
import numpy as np
import pandas as pd
import ccxt
from dataclasses import dataclass
from scipy import stats


# ═══════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════

def fetch_historical_ohlcv(symbol="BTC/USDT", exchange_name="binance",
                            timeframe="1m", days=7, limit_per_call=1000):
    """
    Pulls OHLCV candles from any ccxt-supported exchange via the
    public REST endpoint — no API key needed for market data.
    Paginates backward until `days` of history is collected.
    """
    exchange = getattr(ccxt, exchange_name)()
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    now   = exchange.milliseconds()
    since = now - int(days * 24 * 3600 * 1000)

    all_candles, fetch_since = [], since
    while fetch_since < now:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                      since=fetch_since, limit=limit_per_call)
        if not batch:
            break
        all_candles += batch
        fetch_since  = batch[-1][0] + ms_per_candle
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < limit_per_call:
            break

    df = pd.DataFrame(all_candles,
                      columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").reset_index(drop=True)
    return df


def save_ohlcv_csv(df, path): df.to_csv(path, index=False)
def load_ohlcv_csv(path):     return pd.read_csv(path, parse_dates=["timestamp"])


# ═══════════════════════════════════════════════════════════
# 2. SIGNALS
# ═══════════════════════════════════════════════════════════

def compute_bar_portion(df: pd.DataFrame, lookback=5) -> pd.Series:
    """
    Bar Portion = (Close − Open) / (High − Low)
    Exponentially weighted over `lookback` candles.
    Source: Stoikov et al., Cornell FE Manhattan 2024.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    raw = ((df["close"] - df["open"]) / rng).fillna(0.0)
    w   = np.exp(np.linspace(-1, 0, lookback)); w /= w.sum()
    return raw.rolling(lookback).apply(lambda x: np.dot(w, x), raw=True).fillna(0.0)


# ── plug-in point: add your own signal function here ──────
# def compute_my_signal(df) -> pd.Series: ...
# ──────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════
# 3. RISK & COST MODEL
# ═══════════════════════════════════════════════════════════

@dataclass
class RiskParams:
    # ── transaction costs ──────────────────────────────────
    maker_fee_bps:  float = 2.0   # 0.02 % — typical maker fee
    taker_fee_bps:  float = 4.0   # 0.04 % — typical taker fee
    slippage_bps:   float = 1.0   # extra adverse fill movement
    use_taker:      bool  = False  # market makers post maker orders

    # ── inventory / position risk ──────────────────────────
    max_inventory:      float = 0.05   # hard cap on |inventory| (BTC)
    max_drawdown_halt:  float = 0.20   # circuit breaker: halt at 20 % drawdown
    size_pct_of_equity: float = 0.01   # order notional = 1 % of current equity

    def effective_fee_bps(self):
        return self.taker_fee_bps if self.use_taker else self.maker_fee_bps

    def apply_cost(self, price: float, side: str) -> float:
        """Returns effective fill price after fees + slippage."""
        adj = (self.effective_fee_bps() + self.slippage_bps) / 10_000
        return price * (1 + adj) if side == "buy" else price * (1 - adj)


# ═══════════════════════════════════════════════════════════
# 4. STRATEGIES  (plug-in interface)
# ═══════════════════════════════════════════════════════════

class BaselineInventoryMM:
    """
    Control group — pure Avellaneda-Stoikov inventory-skew quoting,
    no alpha signal. Baseline to beat.
    """
    name = "Baseline (no signal)"

    def __init__(self, gamma=0.1, spread_vol_mult=4.0):
        self.gamma, self.spread_vol_mult = gamma, spread_vol_mult

    def get_quotes(self, mid, inventory, sigma, signal):
        spread   = max(self.spread_vol_mult * sigma * mid, mid * 0.001)
        skew     = inventory * self.gamma * sigma**2 * mid
        res      = mid - skew
        return res - spread/2, res + spread/2, spread


class BarPortionMM:
    """
    Treatment group — Avellaneda-Stoikov + Bar Portion mean-reversion
    alpha. (Stoikov et al. 2024)
    """
    name = "Bar Portion MM"

    def __init__(self, gamma=0.1, alpha_weight=0.3, spread_vol_mult=4.0):
        self.gamma, self.alpha_weight, self.spread_vol_mult = (
            gamma, alpha_weight, spread_vol_mult)

    def get_quotes(self, mid, inventory, sigma, signal):
        spread     = max(self.spread_vol_mult * sigma * mid, mid * 0.001)
        inv_skew   = inventory   * self.gamma        * sigma**2 * mid
        sig_skew   = -signal     * self.alpha_weight * sigma    * mid
        res        = mid - inv_skew + sig_skew
        return res - spread/2, res + spread/2, spread


# ── add your own strategy class here, same interface ──────
# class MyStrategy:
#     name = "My Strategy"
#     def get_quotes(self, mid, inventory, sigma, signal):
#         ...  return bid, ask, spread
# ──────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════
# 5. BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, strategy,
                  risk: RiskParams, signal_series: pd.Series,
                  vol_lookback=20, starting_cash=10_000):
    """
    Vectorised-friendly loop over OHLCV rows.

    Risk controls applied on every bar:
      1. Dynamic order sizing  — notional = risk.size_pct_of_equity * equity / price
      2. Inventory cap         — blocks the side that would breach max_inventory
      3. Circuit breaker       — halts all new orders once drawdown > max_drawdown_halt

    Fill model: next-bar close crosses our quoted price → fill at cost-adjusted price.
    """
    closes  = df["close"].values
    signals = signal_series.values

    cash, inventory    = starting_cash, 0.0
    peak_equity        = starting_cash
    halted             = False

    curve, trades = [], []

    for i in range(vol_lookback, len(df) - 1):
        mid    = closes[i]
        equity = cash + inventory * mid

        # ── update peak and check circuit breaker ─────────
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        if risk.max_drawdown_halt and dd <= -risk.max_drawdown_halt:
            halted = True

        # ── dynamic order size ────────────────────────────
        order_size = max(risk.size_pct_of_equity * equity / mid, 0.0)

        # ── volatility estimate ───────────────────────────
        ph    = closes[max(0, i - vol_lookback): i]
        sigma = float(np.std(np.diff(np.log(ph)))) if len(ph) > 1 else 0.001
        sigma = max(sigma, 1e-6)

        bid, ask, spread = strategy.get_quotes(mid, inventory, sigma, signals[i])
        next_price       = closes[i + 1]

        # ── risk gates ────────────────────────────────────
        can_buy  = (not halted) and (inventory + order_size <=  risk.max_inventory)
        can_sell = (not halted) and (inventory - order_size >= -risk.max_inventory)

        if can_buy and next_price <= bid:
            fp    = risk.apply_cost(bid, "buy")
            cash -= fp * order_size;  inventory += order_size
            trades.append({"bar": i, "side": "buy",  "price": fp, "size": order_size})

        if can_sell and next_price >= ask:
            fp    = risk.apply_cost(ask, "sell")
            cash += fp * order_size;  inventory -= order_size
            trades.append({"bar": i, "side": "sell", "price": fp, "size": order_size})

        equity = cash + inventory * mid
        curve.append({"timestamp": df["timestamp"].iloc[i],
                       "equity": equity, "inventory": inventory,
                       "mid": mid, "drawdown": dd, "halted": halted})

    eq_df           = pd.DataFrame(curve)
    eq_df["returns"] = eq_df["equity"].pct_change().fillna(0.0)
    return eq_df, pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════
# 6. PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════

def performance_metrics(eq_df: pd.DataFrame,
                         starting_cash=10_000,
                         periods_per_year=525_600) -> dict:
    r           = eq_df["returns"]
    final_eq    = eq_df["equity"].iloc[-1]
    total_ret   = (final_eq - starting_cash) / starting_cash
    sharpe      = r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else 0.0
    running_max = eq_df["equity"].cummax()
    max_dd      = ((eq_df["equity"] - running_max) / running_max).min()
    n_trades    = int((eq_df.get("inventory", pd.Series()).diff().abs() > 0).sum())

    return {
        "total_return_%":        round(total_ret * 100, 4),
        "sharpe_ratio":          round(float(sharpe), 4),
        "max_drawdown_%":        round(float(max_dd) * 100, 4),
        "final_equity_$":        round(final_eq, 2),
        "max_inventory_reached": round(eq_df["inventory"].abs().max(), 6),
        "pct_time_halted_%":     round(eq_df["halted"].mean() * 100, 2),
        "n_periods":             len(eq_df),
    }


# ═══════════════════════════════════════════════════════════
# 7. SIGNAL SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════

def signal_significance_test(df: pd.DataFrame,
                               signal: pd.Series,
                               forward_window=1) -> dict:
    """
    Two independent tests:
      1. OLS regression: forward_return ~ signal value
         → p-value + R² (R² ≈ 0 means no practical effect even if p < 0.05)
      2. Quintile monotonicity (paper's own methodology)
         → monotonically decreasing means high BP predicts negative returns
    """
    fwd_ret = df["close"].pct_change(forward_window).shift(-forward_window)
    valid   = (~signal.isna()) & (~fwd_ret.isna())
    x, y    = signal[valid].values, fwd_ret[valid].values

    slope, _, r_val, p_val, _ = stats.linregress(x, y)

    quintiles      = pd.qcut(x, 5, labels=False, duplicates="drop")
    q_means        = pd.Series(y).groupby(quintiles).mean()

    return {
        "n_obs":                 len(x),
        "regression_slope":      round(slope, 8),
        "r_squared":             round(r_val**2, 8),
        "p_value":               round(p_val, 6),
        "significant_at_5pct":   p_val < 0.05,
        "quintile_means":        {int(k): round(v, 8) for k, v in q_means.items()},
        "monotonic_decreasing":  q_means.is_monotonic_decreasing,
        "monotonic_increasing":  q_means.is_monotonic_increasing,
    }


def paired_strategy_test(returns_a: pd.Series,
                           returns_b: pd.Series,
                           n_bootstrap=2000) -> dict:
    """
    Paired t-test + bootstrap 95 % CI on mean return difference (A − B).
    If CI excludes zero AND p < 0.05, strategy A has a statistically
    detectable edge over B.
    """
    diff          = (returns_a.values - returns_b.values)
    t, p          = stats.ttest_rel(returns_a, returns_b)
    rng           = np.random.default_rng(42)
    boot          = [rng.choice(diff, len(diff), replace=True).mean()
                     for _ in range(n_bootstrap)]
    ci_lo, ci_hi  = np.percentile(boot, [2.5, 97.5])

    return {
        "mean_diff_per_period":  diff.mean(),
        "t_stat":                round(float(t), 4),
        "p_value":               round(float(p), 6),
        "significant_at_5pct":   p < 0.05,
        "bootstrap_95ci":        (round(ci_lo, 10), round(ci_hi, 10)),
        "ci_excludes_zero":      not (ci_lo <= 0 <= ci_hi),
    }


# ═══════════════════════════════════════════════════════════
# 8. MAIN COMPARISON RUNNER
# ═══════════════════════════════════════════════════════════

def compare_strategies(df: pd.DataFrame,
                        risk: RiskParams = None,
                        extra_strategies: list = None,
                        starting_cash=10_000) -> dict:
    """
    Runs Baseline + BarPortion (+ any extra_strategies you pass in)
    through the same backtest engine and risk parameters and returns
    a full results dict.

    extra_strategies: list of strategy instances implementing get_quotes().
    """
    risk    = risk or RiskParams()
    signal  = compute_bar_portion(df)

    strategies = [BaselineInventoryMM(), BarPortionMM()]
    if extra_strategies:
        strategies += extra_strategies

    all_results = {}
    eq_frames   = {}

    for strat in strategies:
        eq, trades = run_backtest(df, strat, risk, signal,
                                   starting_cash=starting_cash)
        all_results[strat.name] = {
            "metrics": performance_metrics(eq, starting_cash=starting_cash),
            "eq_df":   eq,
            "trades":  trades,
        }
        eq_frames[strat.name] = eq

    sig_test = signal_significance_test(df, signal)

    # paired comparison: every strategy vs the baseline
    baseline_ret = eq_frames[BaselineInventoryMM.name]["returns"]
    paired = {}
    for name, res in all_results.items():
        if name == BaselineInventoryMM.name:
            continue
        n = min(len(baseline_ret), len(res["eq_df"]))
        paired[name] = paired_strategy_test(
            res["eq_df"]["returns"].iloc[:n],
            baseline_ret.iloc[:n],
        )

    return {
        "risk_params":        risk,
        "strategies":         all_results,
        "signal_significance": sig_test,
        "vs_baseline":        paired,
    }


# ═══════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    days      = int(sys.argv[1])      if len(sys.argv) > 1 else 7
    timeframe = sys.argv[2]           if len(sys.argv) > 2 else "1m"
    symbol    = sys.argv[3]           if len(sys.argv) > 3 else "BTC/USDT"

    print(f"\nFetching {days}d of {timeframe} candles for {symbol} from Binance...")
    df = fetch_historical_ohlcv(symbol=symbol, timeframe=timeframe, days=days)
    span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
    print(f"Got {len(df):,} candles  ({span:.2f} days actual span)\n")

    if span < days * 0.85:
        print("  ⚠ WARNING: got significantly fewer candles than requested.\n"
              "    Try a shorter window or coarser timeframe (e.g. 15m).\n")

    save_ohlcv_csv(df, f"{symbol.replace('/','_')}_{timeframe}.csv")

    risk    = RiskParams(maker_fee_bps=2.0, slippage_bps=1.0,
                         max_inventory=0.05, max_drawdown_halt=0.20,
                         size_pct_of_equity=0.01)
    results = compare_strategies(df, risk=risk)

    # ── print metrics ──────────────────────────────────────
    for name, res in results["strategies"].items():
        print(f"{'═'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        for k, v in res["metrics"].items():
            print(f"  {k:35s}: {v}")

    print(f"\n{'═'*50}")
    print("  SIGNAL SIGNIFICANCE  (Bar Portion → fwd return)")
    print(f"{'─'*50}")
    sig = results["signal_significance"]
    print(f"  n_obs                              : {sig['n_obs']:,}")
    print(f"  regression p-value                 : {sig['p_value']:.6f}  "
          f"({'✓ significant' if sig['significant_at_5pct'] else '✗ not significant'} at 5 %)")
    print(f"  R-squared                          : {sig['r_squared']:.8f}")
    print(f"  quintile monotonic decreasing      : {sig['monotonic_decreasing']}")
    print(f"  quintile means                     : "
          + "  ".join(f"Q{k}={v:.6f}" for k,v in sig["quintile_means"].items()))

    print(f"\n{'═'*50}")
    print("  STRATEGY vs BASELINE  (paired test)")
    print(f"{'─'*50}")
    for name, ps in results["vs_baseline"].items():
        print(f"  [{name}]")
        print(f"  mean return diff / period          : {ps['mean_diff_per_period']:.10f}")
        print(f"  p-value                            : {ps['p_value']:.6f}  "
              f"({'✓' if ps['significant_at_5pct'] else '✗'} at 5 %)")
        print(f"  bootstrap 95 % CI                  : {ps['bootstrap_95ci']}")
        print(f"  CI excludes zero (real edge)       : {ps['ci_excludes_zero']}")
