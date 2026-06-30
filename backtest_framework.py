# ═══════════════════════════════════════════════════════════
# BACKTEST FRAMEWORK
# Compares a Bar-Portion-signal market maker against a
# no-signal baseline (pure inventory-skew quoting), with
# transaction costs + slippage, and tests whether the Bar
# Portion signal has real statistical significance.
#
# Data: pulled via ccxt's public fetch_ohlcv (no API key
# needed for historical OHLCV on most exchanges).
# ═══════════════════════════════════════════════════════════
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
    Pulls historical OHLCV candles via ccxt's public REST endpoint
    (no API key required). Paginates backward in time until `days`
    of history is collected.
    """
    exchange = getattr(ccxt, exchange_name)()
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    now = exchange.milliseconds()
    since = now - days * 24 * 60 * 60 * 1000

    all_candles = []
    fetch_since = since
    while fetch_since < now:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                      since=fetch_since, limit=limit_per_call)
        if not batch:
            break
        all_candles += batch
        fetch_since = batch[-1][0] + ms_per_candle
        time.sleep(exchange.rateLimit / 1000)  # respect rate limits
        if len(batch) < limit_per_call:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def save_ohlcv_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

# ═══════════════════════════════════════════════════════════
# 2. SIGNAL — BAR PORTION
# ═══════════════════════════════════════════════════════════

def compute_bar_portion(df: pd.DataFrame, lookback=5) -> pd.Series:
    """
    Bar Portion = (Close - Open) / (High - Low), exponentially
    weighted average over `lookback` candles (recent weighted more).
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    raw = (df["close"] - df["open"]) / rng
    raw = raw.fillna(0.0)

    weights = np.exp(np.linspace(-1, 0, lookback))
    weights /= weights.sum()

    bp = raw.rolling(lookback).apply(
        lambda x: np.dot(weights, x), raw=True
    )
    return bp.fillna(0.0)

# ═══════════════════════════════════════════════════════════
# 3. TRANSACTION COST / SLIPPAGE MODEL
# ═══════════════════════════════════════════════════════════

@dataclass
class CostModel:
    maker_fee_bps: float = 2.0      # 0.02% (typical maker fee, matches paper's live setup)
    taker_fee_bps: float = 4.0      # 0.04% (typical taker fee)
    slippage_bps: float = 1.0       # extra adverse price movement on fill
    use_taker: bool = False         # market makers usually post maker orders

    def fee_bps(self) -> float:
        return self.taker_fee_bps if self.use_taker else self.maker_fee_bps

    def apply_cost(self, fill_price: float, side: str) -> float:
        """
        Returns the effective fill price after fees + slippage.
        Buys get worse (higher) effective price; sells get worse (lower).
        """
        total_bps = self.fee_bps() + self.slippage_bps
        adj = total_bps / 10_000
        if side == "buy":
            return fill_price * (1 + adj)
        else:
            return fill_price * (1 - adj)

# ═══════════════════════════════════════════════════════════
# 4. QUOTE STRATEGIES
# ═══════════════════════════════════════════════════════════

class BaselineInventoryMM:
    """No-signal pure inventory-skew market maker (Avellaneda-Stoikov style,
    no Bar Portion alpha — this is the control group)."""
    name = "Baseline (no signal)"

    def __init__(self, gamma=0.1, spread_vol_mult=4.0, order_size=0.01):
        self.gamma = gamma
        self.spread_vol_mult = spread_vol_mult
        self.order_size = order_size

    def get_quotes(self, mid, inventory, sigma, bar_portion_unused):
        spread = max(self.spread_vol_mult * sigma * mid, mid * 0.001)
        inventory_skew = inventory * self.gamma * sigma**2 * mid
        reservation = mid - inventory_skew
        return reservation - spread / 2, reservation + spread / 2, spread


class BarPortionMM:
    """Bar Portion alpha-augmented market maker (treatment group)."""
    name = "Bar Portion MM"

    def __init__(self, gamma=0.1, alpha_weight=0.3, spread_vol_mult=4.0, order_size=0.01):
        self.gamma = gamma
        self.alpha_weight = alpha_weight
        self.spread_vol_mult = spread_vol_mult
        self.order_size = order_size

    def get_quotes(self, mid, inventory, sigma, bar_portion):
        spread = max(self.spread_vol_mult * sigma * mid, mid * 0.001)
        inventory_skew = inventory * self.gamma * sigma**2 * mid
        signal_skew = -bar_portion * self.alpha_weight * sigma * mid
        reservation = mid - inventory_skew + signal_skew
        return reservation - spread / 2, reservation + spread / 2, spread

# ═══════════════════════════════════════════════════════════
# 5. BACKTEST ENGINE (with costs + slippage)
# ═══════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, strategy, cost_model: CostModel,
                  order_size=0.01, vol_lookback=20, starting_cash=10_000):
    closes = df["close"].values
    bar_portions = compute_bar_portion(df).values

    cash = starting_cash
    inventory = 0.0
    equity_curve = []
    trade_log = []

    for i in range(vol_lookback, len(df) - 1):
        price_hist = closes[max(0, i - vol_lookback):i]
        sigma = float(np.std(np.diff(np.log(price_hist)))) if len(price_hist) > 1 else 0.001
        sigma = max(sigma, 1e-6)

        mid = closes[i]
        bp = bar_portions[i]

        bid, ask, spread = strategy.get_quotes(mid, inventory, sigma, bp)

        next_price = closes[i + 1]

        # simulate fill: did next tick's price cross our quote?
        if next_price <= bid:
            fill_price = cost_model.apply_cost(bid, "buy")
            cash -= fill_price * order_size
            inventory += order_size
            trade_log.append({"i": i, "side": "buy", "price": fill_price})
        if next_price >= ask:
            fill_price = cost_model.apply_cost(ask, "sell")
            cash += fill_price * order_size
            inventory -= order_size
            trade_log.append({"i": i, "side": "sell", "price": fill_price})

        equity = cash + inventory * mid
        equity_curve.append({
            "timestamp": df["timestamp"].iloc[i],
            "equity": equity,
            "inventory": inventory,
            "mid": mid,
        })

    eq_df = pd.DataFrame(equity_curve)
    eq_df["returns"] = eq_df["equity"].pct_change().fillna(0.0)
    return eq_df, pd.DataFrame(trade_log)


def performance_metrics(eq_df: pd.DataFrame, starting_cash=10_000, periods_per_year=525_600):
    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity - starting_cash) / starting_cash

    returns = eq_df["returns"]
    sharpe = (returns.mean() / returns.std() * np.sqrt(periods_per_year)
              if returns.std() > 0 else 0.0)

    running_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "total_return_pct": round(total_return * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "final_equity": round(final_equity, 2),
        "n_periods": len(eq_df),
    }

# ═══════════════════════════════════════════════════════════
# 6. STATISTICAL SIGNIFICANCE OF THE BAR PORTION SIGNAL
# ═══════════════════════════════════════════════════════════

def signal_significance_test(df: pd.DataFrame, lookback=5, forward_window=1):
    """
    Tests whether Bar Portion has real predictive power over forward returns,
    independent of any specific strategy. Two checks:
      1. Regression t-stat: forward_return ~ bar_portion
      2. Quintile monotonicity, replicating the paper's methodology
    """
    bp = compute_bar_portion(df, lookback=lookback)
    fwd_ret = df["close"].pct_change(forward_window).shift(-forward_window)

    valid = (~bp.isna()) & (~fwd_ret.isna())
    x = bp[valid].values
    y = fwd_ret[valid].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # quintile analysis
    quintiles = pd.qcut(x, 5, labels=False, duplicates="drop")
    quintile_means = pd.Series(y).groupby(quintiles).mean()
    is_monotonic_dec = quintile_means.is_monotonic_decreasing
    is_monotonic_inc = quintile_means.is_monotonic_increasing

    return {
        "n_obs": len(x),
        "regression_slope": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
        "quintile_means": quintile_means.to_dict(),
        "monotonic_decreasing": is_monotonic_dec,
        "monotonic_increasing": is_monotonic_inc,
    }


def paired_strategy_significance(returns_a: pd.Series, returns_b: pd.Series, n_bootstrap=2000):
    """
    Tests whether strategy A's per-period returns are significantly
    different from strategy B's, via both a paired t-test and a
    bootstrap confidence interval on the mean difference.
    """
    diff = returns_a.values - returns_b.values
    t_stat, p_value = stats.ttest_rel(returns_a, returns_b)

    rng = np.random.default_rng(42)
    boot_means = []
    n = len(diff)
    for _ in range(n_bootstrap):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean_diff": diff.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "significant_at_5pct": p_value < 0.05,
        "bootstrap_95ci": (ci_low, ci_high),
        "ci_excludes_zero": not (ci_low <= 0 <= ci_high),
    }

# ═══════════════════════════════════════════════════════════
# 7. END-TO-END COMPARISON RUNNER
# ═══════════════════════════════════════════════════════════

def compare_strategies(df: pd.DataFrame, cost_model: CostModel = None):
    cost_model = cost_model or CostModel()

    baseline = BaselineInventoryMM()
    bar_portion_strat = BarPortionMM()

    eq_baseline, trades_baseline = run_backtest(df, baseline, cost_model)
    eq_bp, trades_bp = run_backtest(df, bar_portion_strat, cost_model)

    metrics_baseline = performance_metrics(eq_baseline)
    metrics_bp = performance_metrics(eq_bp)

    sig_test = signal_significance_test(df)

    # align lengths for paired comparison (both loops run same range)
    min_len = min(len(eq_baseline), len(eq_bp))
    strat_sig = paired_strategy_significance(
        eq_bp["returns"].iloc[:min_len],
        eq_baseline["returns"].iloc[:min_len],
    )

    return {
        "baseline_metrics": metrics_baseline,
        "bar_portion_metrics": metrics_bp,
        "signal_significance": sig_test,
        "strategy_vs_baseline_significance": strat_sig,
        "eq_baseline": eq_baseline,
        "eq_bp": eq_bp,
        "trades_baseline": trades_baseline,
        "trades_bp": trades_bp,
    }


# ═══════════════════════════════════════════════════════════
# 8. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Fetching historical data (BTC/USDT, 1m candles, last 3 days)...")
    df = fetch_historical_ohlcv(symbol="BTC/USDT", exchange_name="binance",
                                 timeframe="1m", days=3)
    print(f"Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

    save_ohlcv_csv(df, "btc_usdt_1m.csv")

    cost_model = CostModel(maker_fee_bps=2.0, slippage_bps=1.0)
    results = compare_strategies(df, cost_model)

    print("\n=== BASELINE (no signal) ===")
    for k, v in results["baseline_metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== BAR PORTION MM ===")
    for k, v in results["bar_portion_metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== SIGNAL SIGNIFICANCE (Bar Portion vs forward returns) ===")
    sig = results["signal_significance"]
    print(f"  n_obs: {sig['n_obs']}")
    print(f"  regression p-value: {sig['p_value']:.6f}  (significant at 5%: {sig['significant_at_5pct']})")
    print(f"  R-squared: {sig['r_squared']:.6f}")
    print(f"  monotonic decreasing across quintiles: {sig['monotonic_decreasing']}")

    print("\n=== STRATEGY vs BASELINE SIGNIFICANCE (paired test on returns) ===")
    strat_sig = results["strategy_vs_baseline_significance"]
    print(f"  mean return diff per period: {strat_sig['mean_diff']:.8f}")
    print(f"  p-value: {strat_sig['p_value']:.6f}  (significant at 5%: {strat_sig['significant_at_5pct']})")
    print(f"  bootstrap 95% CI on diff: {strat_sig['bootstrap_95ci']}")
    print(f"  CI excludes zero (real edge): {strat_sig['ci_excludes_zero']}")
