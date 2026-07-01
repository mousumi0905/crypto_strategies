# ═══════════════════════════════════════════════════════════
# PMM-BP LIVE PAPER TRADING DASHBOARD  v2.0
# ─────────────────────────────────────────────────────────
# Risk controls added (v2):
#   • Max inventory cap — blocks quotes that breach limit
#   • Circuit breaker  — halts quoting past max drawdown
#   • Dynamic sizing   — order size = pct of equity / price
#   • Fee-adjusted PnL — every simulated fill deducts fees
#   • Risk status panel in dashboard
# ═══════════════════════════════════════════════════════════
import asyncio
import threading
import time
import numpy as np
import pandas as pd
import streamlit as st
import ccxt
import ccxt.pro as ccxtpro
from dataclasses import dataclass, field
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="PMM-BP Market Maker v2")

# ═══════════════════════════════════════════════════════════
# SHARED STATE  — created once via cache_resource so it
# survives every Streamlit rerun (fixes the "running resets
# to False" bug from v1).
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def get_shared_state():
    return {
        # market data
        "running":        False,
        "mid_price":      None,
        "best_bid":       None,
        "best_ask":       None,
        "last_price":     None,
        "orderbook":      {"bids": [], "asks": []},
        "price_history":  [],
        "tick_buffer":    [],
        "latest_candle":  None,
        # strategy
        "current_quote":  None,
        "active_orders":  [],
        "quote_history":  [],
        "signal_str":     "Waiting...",
        "bar_portion":    0.0,
        # risk & pnl
        "inventory":      0.0,
        "cash":           10_000.0,
        "peak_equity":    10_000.0,
        "pnl":            0.0,
        "halted":         False,
        "total_fees_paid": 0.0,
        # errors
        "ws_error":       None,
        "strategy_error": None,
    }

SS = get_shared_state()   # short alias used everywhere below

# ═══════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════
@dataclass
class Candle:
    open: float; high: float; low: float; close: float; volume: float

@dataclass
class Quote:
    bid_price: float; ask_price: float
    bid_size:  float; ask_size:  float
    signal:    float; spread:    float

@dataclass
class Order:
    order_id:  str
    side:      str
    price:     float
    size:      float
    status:    str
    timestamp: float = field(default_factory=time.time)

# ═══════════════════════════════════════════════════════════
# RISK PARAMETERS  (dataclass so it's easy to pass around)
# ═══════════════════════════════════════════════════════════
@dataclass
class RiskParams:
    maker_fee_bps:      float = 2.0
    slippage_bps:       float = 1.0
    max_inventory:      float = 0.05
    max_drawdown_halt:  float = 0.20
    size_pct_of_equity: float = 0.01

    def apply_cost(self, price: float, side: str) -> float:
        adj = (self.maker_fee_bps + self.slippage_bps) / 10_000
        return price * (1 + adj) if side == "buy" else price * (1 - adj)

    def fee_cost(self, price: float, size: float) -> float:
        return price * size * (self.maker_fee_bps + self.slippage_bps) / 10_000

# ═══════════════════════════════════════════════════════════
# BAR PORTION SIGNAL
# ═══════════════════════════════════════════════════════════
class BarPortionSignal:
    def __init__(self, lookback=5):
        self.lookback = lookback
        self.history  = []

    def update(self, candle: Candle):
        self.history.append(candle)
        if len(self.history) > self.lookback:
            self.history.pop(0)

    def compute(self) -> float:
        if not self.history:
            return 0.0
        vals = []
        for c in self.history:
            r = c.high - c.low
            vals.append(0.0 if r == 0 else (c.close - c.open) / r)
        w = np.exp(np.linspace(-1, 0, len(vals))); w /= w.sum()
        return float(np.dot(w, vals))

    def label(self) -> str:
        s = self.compute()
        if s >  0.5: return "🔴 Strong Bull → Expect Reversal DOWN"
        if s >  0.2: return "🟠 Mild Bull → Slight Reversal"
        if s < -0.5: return "🟢 Strong Bear → Expect Reversal UP"
        if s < -0.2: return "🟡 Mild Bear → Slight Reversal"
        return "⚪ Neutral"

# ═══════════════════════════════════════════════════════════
# QUOTE CALCULATOR
# ═══════════════════════════════════════════════════════════
class QuoteCalculator:
    def __init__(self, gamma=0.1, alpha_weight=0.3, spread_vol_mult=4.0):
        self.gamma, self.alpha_weight, self.svm = gamma, alpha_weight, spread_vol_mult

    def _vol(self, prices):
        if len(prices) < 2: return 0.001
        r = np.diff(np.log(prices[-20:]))
        return max(float(np.std(r)), 1e-6)

    def get_quotes(self, mid, inventory, bar_portion,
                   price_history, order_size) -> Quote:
        sigma  = self._vol(price_history)
        spread = max(self.svm * sigma * mid, mid * 0.001)
        inv_sk = inventory * self.gamma * sigma**2 * mid
        sig_sk = -bar_portion * self.alpha_weight * sigma * mid
        res    = mid - inv_sk + sig_sk
        return Quote(
            bid_price = round(res - spread/2, 4),
            ask_price = round(res + spread/2, 4),
            bid_size  = order_size,
            ask_size  = order_size,
            signal    = bar_portion,
            spread    = round(spread, 6),
        )

# ═══════════════════════════════════════════════════════════
# ORDER MANAGER  (demo mode — no real orders)
# ═══════════════════════════════════════════════════════════
class OrderManager:
    def __init__(self, demo_mode=True):
        self.demo_mode     = demo_mode
        self.active_orders = {}
        self._counter      = 0

    def place(self, side, price, size) -> Order:
        self._counter += 1
        o = Order(f"demo_{self._counter}", side, price, size, "open")
        self.active_orders[o.order_id] = o
        return o

    def cancel_all(self):
        self.active_orders.clear()

    def simulate_fills(self, current_price, cash, inventory, risk: RiskParams):
        """
        Check each open order against current price.
        Deducts fees on every fill.
        Returns updated (cash, inventory, fees_paid).
        """
        fees = 0.0
        for oid, o in list(self.active_orders.items()):
            if o.side == "buy" and current_price <= o.price:
                fp    = risk.apply_cost(o.price, "buy")
                fee   = risk.fee_cost(o.price, o.size)
                cash -= fp * o.size
                inventory += o.size
                fees  += fee
                del self.active_orders[oid]
            elif o.side == "sell" and current_price >= o.price:
                fp    = risk.apply_cost(o.price, "sell")
                fee   = risk.fee_cost(o.price, o.size)
                cash += fp * o.size
                inventory -= o.size
                fees  += fee
                del self.active_orders[oid]
        return cash, inventory, fees

# ═══════════════════════════════════════════════════════════
# WEBSOCKET FEED
# ═══════════════════════════════════════════════════════════
async def stream_market_data(symbol, exchange_name, state):
    try:
        exchange = getattr(ccxtpro, exchange_name)()
    except Exception as e:
        import traceback
        state["ws_error"] = f"Exchange init failed: {e}\n{traceback.format_exc()}"
        state["running"]  = False
        return

    print(f"[WS] Connecting to {exchange_name} for {symbol}...")
    try:
        while state["running"]:
            try:
                ob, ticker = await asyncio.gather(
                    exchange.watch_order_book(symbol, limit=5),
                    exchange.watch_ticker(symbol),
                )
                if ob["bids"] and ob["asks"]:
                    state["best_bid"]  = ob["bids"][0][0]
                    state["best_ask"]  = ob["asks"][0][0]
                    state["mid_price"] = (state["best_bid"] + state["best_ask"]) / 2
                    state["orderbook"] = {"bids": ob["bids"][:5], "asks": ob["asks"][:5]}

                state["last_price"] = ticker["last"]
                state["price_history"].append(ticker["last"])
                if len(state["price_history"]) > 200:
                    state["price_history"].pop(0)

                state["tick_buffer"].append(ticker["last"])
                if len(state["tick_buffer"]) >= 60:
                    p = state["tick_buffer"]
                    state["latest_candle"] = Candle(p[0], max(p), min(p), p[-1],
                                                     ticker.get("baseVolume", 0))
                    state["tick_buffer"] = []
            except Exception as e:
                state["ws_error"] = f"{type(e).__name__}: {e}"
                await asyncio.sleep(3)
    finally:
        await exchange.close()


def start_ws_thread(symbol, exchange_name, state):
    if st.session_state.get("ws_started"):
        return
    st.session_state.ws_started = True
    state["running"] = True

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(stream_market_data(symbol, exchange_name, state))
        except Exception as e:
            import traceback
            state["ws_error"] = f"{e}\n{traceback.format_exc()}"
            state["running"]  = False

    threading.Thread(target=_run, daemon=True).start()
    print("[WS] Thread launched")

# ═══════════════════════════════════════════════════════════
# STRATEGY LOOP
# ═══════════════════════════════════════════════════════════
def strategy_loop(signal_eng, quote_calc, order_mgr,
                   risk: RiskParams, state, refresh_secs):
    print("[STRATEGY] Loop started")
    try:
        while state["running"]:
            mid = state.get("mid_price")
            if mid is None:
                time.sleep(1); continue

            candle = state.get("latest_candle")
            if candle:
                signal_eng.update(candle)

            bp           = signal_eng.compute()
            price_hist   = state.get("price_history", [mid])
            cash         = state.get("cash", 10_000.0)
            inventory    = state.get("inventory", 0.0)
            peak_equity  = state.get("peak_equity", 10_000.0)
            equity       = cash + inventory * mid

            # ── circuit breaker check ─────────────────────
            peak_equity  = max(peak_equity, equity)
            dd           = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
            halted       = dd <= -risk.max_drawdown_halt
            state["halted"]      = halted
            state["peak_equity"] = peak_equity

            if halted:
                state["signal_str"]   = "🛑 HALTED — max drawdown breached"
                state["bar_portion"]  = bp
                state["active_orders"] = list(order_mgr.active_orders.values())
                time.sleep(refresh_secs); continue

            # ── dynamic order size ────────────────────────
            order_size = max(risk.size_pct_of_equity * equity / mid, 0.0)

            # ── inventory gate ────────────────────────────
            at_long_cap  = inventory + order_size >  risk.max_inventory
            at_short_cap = inventory - order_size < -risk.max_inventory

            quote = quote_calc.get_quotes(mid, inventory, bp,
                                           price_hist, order_size)

            # ── place orders (only sides that aren't capped) ─
            order_mgr.cancel_all()
            if not at_long_cap:
                order_mgr.place("buy",  quote.bid_price, order_size)
            if not at_short_cap:
                order_mgr.place("sell", quote.ask_price, order_size)

            # ── simulate fills ────────────────────────────
            cash, inventory, fees = order_mgr.simulate_fills(
                mid, cash, inventory, risk)

            equity = cash + inventory * mid
            state["cash"]            = cash
            state["inventory"]       = inventory
            state["pnl"]             = equity - 10_000.0
            state["total_fees_paid"] = state.get("total_fees_paid", 0.0) + fees
            state["current_quote"]   = quote
            state["signal_str"]      = signal_eng.label()
            state["bar_portion"]     = bp
            state["active_orders"]   = list(order_mgr.active_orders.values())

            hist = state.get("quote_history", [])
            hist.append({"time": time.time(), "mid": mid,
                          "bid": quote.bid_price, "ask": quote.ask_price,
                          "signal": bp, "inventory": inventory, "equity": equity})
            if len(hist) > 200: hist.pop(0)
            state["quote_history"] = hist

            time.sleep(refresh_secs)

    except Exception as e:
        import traceback
        err = f"{e}\n{traceback.format_exc()}"
        print(f"[STRATEGY] crashed: {err}")
        state["strategy_error"] = err

# ═══════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════
def init_session():
    for k, v in {"mm_running": False, "ws_started": False, "symbol": "BTC/USDT"}.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    st.title("📊 PMM-BP Market Maker  v2.0")
    st.caption("Avellaneda-Stoikov + Bar Portion alpha · "
               "Stoikov et al., Cornell FE Manhattan 2024")

    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        exchange_name = st.selectbox("Exchange", ["binance", "kraken"])
        symbol        = st.text_input("Symbol", "BTC/USDT")
        refresh_secs  = st.slider("Quote Refresh (sec)", 10, 300, 60)

        st.divider()
        st.header("Strategy Parameters")
        gamma        = st.slider("Risk Aversion (γ)", 0.01, 0.5, 0.1)
        alpha_weight = st.slider("Signal Weight (α)", 0.0, 1.0, 0.3)
        spread_mult  = st.slider("Spread Multiplier", 1.0, 8.0, 4.0)

        st.divider()
        st.header("🛡️ Risk Controls")
        max_inv    = st.number_input("Max Inventory (BTC)",    value=0.05, step=0.01)
        max_dd     = st.slider("Circuit Breaker Drawdown %",   5, 50, 20) / 100
        size_pct   = st.slider("Order Size (% of equity)",     0.1, 5.0, 1.0) / 100
        maker_fee  = st.number_input("Maker Fee (bps)",        value=2.0, step=0.5)
        slippage   = st.number_input("Slippage (bps)",         value=1.0, step=0.5)

        st.divider()
        demo_mode  = st.toggle("Demo Mode (no real orders)", value=True)

        c1, c2 = st.columns(2)
        start_btn = c1.button("▶ Start", type="primary",
                               disabled=st.session_state.mm_running)
        stop_btn  = c2.button("⏹ Stop",
                               disabled=not st.session_state.mm_running)

    # ── START ─────────────────────────────────────────────
    if start_btn:
        st.session_state.mm_running = True
        st.session_state.symbol     = symbol
        SS["ws_error"] = SS["strategy_error"] = None

        risk       = RiskParams(maker_fee_bps=maker_fee, slippage_bps=slippage,
                                 max_inventory=max_inv, max_drawdown_halt=max_dd,
                                 size_pct_of_equity=size_pct)
        signal_eng = BarPortionSignal(lookback=5)
        quote_calc = QuoteCalculator(gamma=gamma, alpha_weight=alpha_weight,
                                      spread_vol_mult=spread_mult)
        order_mgr  = OrderManager(demo_mode=demo_mode)

        start_ws_thread(symbol, exchange_name, SS)
        threading.Thread(
            target=strategy_loop,
            args=(signal_eng, quote_calc, order_mgr, risk, SS, refresh_secs),
            daemon=True,
        ).start()
        st.success("✅ Market maker started!")

    # ── STOP ──────────────────────────────────────────────
    if stop_btn:
        st.session_state.mm_running = False
        st.session_state.ws_started = False
        SS["running"] = False
        st.warning("⏹ Stopped.")

    st_autorefresh(interval=1000, key="ui_refresh")

    # ── DEBUG ─────────────────────────────────────────────
    with st.expander("🔍 Debug", expanded=False):
        st.write({k: SS[k] for k in
                  ["running","last_price","best_bid","best_ask","mid_price","halted",
                   "ws_error","strategy_error"]})

    st.divider()

    # ── ROW 1: LIVE METRICS ───────────────────────────────
    st.subheader("📡 Live Market Data")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    lp  = SS["last_price"];  bb = SS["best_bid"];  ba = SS["best_ask"]
    inv = SS["inventory"];   pnl = SS["pnl"];      fees = SS["total_fees_paid"]

    m1.metric("Last Price",  f"${lp:,.4f}"  if lp  else "⏳")
    m2.metric("Best Bid",    f"${bb:,.4f}"  if bb  else "⏳")
    m3.metric("Best Ask",    f"${ba:,.4f}"  if ba  else "⏳")
    m4.metric("Inventory",   f"{inv:.6f}")
    m5.metric("Net PnL",     f"${pnl:.4f}", delta=f"${pnl:.4f}")
    m6.metric("Fees Paid",   f"${fees:.4f}")

    st.divider()

    # ── ROW 2: RISK STATUS ────────────────────────────────
    st.subheader("🛡️ Risk Status")
    r1,r2,r3,r4 = st.columns(4)
    cash         = SS.get("cash", 10_000.0)
    mid          = SS["mid_price"] or 0
    equity       = cash + inv * mid
    peak         = SS.get("peak_equity", 10_000.0)
    live_dd      = (equity - peak) / peak * 100 if peak > 0 else 0.0
    halted       = SS.get("halted", False)

    r1.metric("Equity",        f"${equity:,.2f}")
    r2.metric("Live Drawdown", f"{live_dd:.2f}%",
              delta=f"{live_dd:.2f}%", delta_color="inverse")
    r3.metric("Inventory |x|", f"{abs(inv):.6f}")
    r4.metric("Circuit Breaker",
              "🛑 HALTED" if halted else "✅ Active",
              delta="HALTED" if halted else "OK",
              delta_color="inverse" if halted else "normal")

    if halted:
        st.error("🛑 Circuit breaker tripped — max drawdown exceeded. "
                 "Stop and review before restarting.")

    st.divider()

    # ── ROW 3: SIGNAL + QUOTES + ORDERS ──────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("📈 Bar Portion Signal")
        bp = SS["bar_portion"]
        st.metric("Bar Portion", f"{bp:.4f}")
        st.info(SS["signal_str"])
        st.progress(float(np.clip((bp+1)/2, 0, 1)),
                    text=f"{bp:.3f}  (−1=Bear | 0=Neutral | +1=Bull)")

    with c2:
        st.subheader("💬 Our Quotes")
        q = SS["current_quote"]
        if q:
            qa, qb = st.columns(2)
            qa.metric("BID", f"${q.bid_price:,.4f}", delta="we buy")
            qb.metric("ASK", f"${q.ask_price:,.4f}", delta="we sell")
            st.metric("Spread", f"${q.spread:.6f}")
            if mid:
                st.caption(f"Mid ${mid:,.4f} | "
                           f"Bid offset ${mid-q.bid_price:.4f} | "
                           f"Ask offset ${q.ask_price-mid:.4f}")
        else:
            st.info("⏳ Waiting for first quote cycle...")

    with c3:
        st.subheader("📋 Active Orders")
        orders = SS["active_orders"]
        if orders:
            st.dataframe(pd.DataFrame([{
                "ID": o.order_id[-6:], "Side": o.side.upper(),
                "Price": f"${o.price:,.4f}", "Size": f"{o.size:.6f}",
            } for o in orders]), use_container_width=True, hide_index=True)
        else:
            st.info("No active orders")

    st.divider()

    # ── ROW 4: ORDER BOOK + EQUITY CHART ──────────────────
    co, cc = st.columns([1, 2])

    with co:
        st.subheader("📖 Order Book")
        ob = SS["orderbook"]
        if ob["bids"] and ob["asks"]:
            asks = pd.DataFrame(ob["asks"][:5], columns=["Price","Size"])
            bids = pd.DataFrame(ob["bids"][:5], columns=["Price","Size"])
            asks["Price"] = asks["Price"].apply(lambda x: f"${x:,.4f}")
            bids["Price"] = bids["Price"].apply(lambda x: f"${x:,.4f}")
            asks["Side"]  = "🔴 ASK"; bids["Side"] = "🟢 BID"
            st.dataframe(asks.iloc[::-1], use_container_width=True, hide_index=True)
            st.markdown("<center>─── spread ───</center>", unsafe_allow_html=True)
            st.dataframe(bids, use_container_width=True, hide_index=True)
        else:
            st.warning("⏳ Waiting for order book...")

    with cc:
        st.subheader("📉 Equity & Quote History")
        hist = SS["quote_history"]
        if len(hist) > 2:
            hdf = pd.DataFrame(hist)
            hdf["time"] = pd.to_datetime(hdf["time"], unit="s")
            tab1, tab2 = st.tabs(["Equity", "Mid / Bid / Ask"])
            with tab1:
                st.line_chart(hdf.set_index("time")[["equity"]])
            with tab2:
                st.line_chart(hdf.set_index("time")[["mid","bid","ask"]])
        else:
            st.info("⏳ Collecting history...")

    st.divider()

    # ── ROW 5: SIGNAL + INVENTORY ─────────────────────────
    cs, ci = st.columns(2)
    with cs:
        st.subheader("📊 Signal Over Time")
        if len(SS["quote_history"]) > 2:
            sd = pd.DataFrame(SS["quote_history"])[["time","signal"]]
            sd["time"] = pd.to_datetime(sd["time"], unit="s")
            st.line_chart(sd.set_index("time"))

    with ci:
        st.subheader("📦 Inventory Over Time")
        if len(SS["quote_history"]) > 2:
            id_ = pd.DataFrame(SS["quote_history"])[["time","inventory"]]
            id_["time"] = pd.to_datetime(id_["time"], unit="s")
            st.line_chart(id_.set_index("time"))


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
init_session()
render()
