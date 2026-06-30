# ═══════════════════════════════════════════════════════════
# IMPORTS
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

st.set_page_config(layout="wide", page_title="PMM BP Market Maker")

# ═══════════════════════════════════════════════════════════
# GLOBAL SHARED STATE
# ── Background threads CANNOT write to st.session_state ──
# ── st.cache_resource ensures this dict is created ONCE per
#    server process and the SAME object is reused across every
#    Streamlit rerun (script re-execution). Without this, a
#    plain module-level dict gets recreated on every rerun,
#    which silently resets "running" to False and kills the
#    background threads almost immediately.
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def get_shared_state():
    return {
        "running":       False,
        "mid_price":     None,
        "best_bid":      None,
        "best_ask":      None,
        "last_price":    None,
        "orderbook":     {"bids": [], "asks": []},
        "price_history": [],
        "tick_buffer":   [],
        "latest_candle": None,
        "current_quote": None,
        "active_orders": [],
        "quote_history": [],
        "signal_str":    "Waiting...",
        "bar_portion":   0.0,
        "inventory":     0.0,
        "pnl":           0.0,
    }

SHARED_STATE = get_shared_state()

# ═══════════════════════════════════════════════════════════
# 1. DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class Candle:
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

@dataclass
class Quote:
    bid_price: float
    ask_price: float
    bid_size:  float
    ask_size:  float
    signal:    float
    spread:    float

@dataclass
class Order:
    order_id:  str
    side:      str
    price:     float
    size:      float
    status:    str
    timestamp: float = field(default_factory=time.time)

# ═══════════════════════════════════════════════════════════
# 2. BAR PORTION SIGNAL
# ═══════════════════════════════════════════════════════════

class BarPortionSignal:
    """
    Bar Portion = (Close - Open) / (High - Low)
    +1 = full bull candle → expect MEAN REVERSION down
    -1 = full bear candle → expect MEAN REVERSION up
    """
    def __init__(self, lookback=5):
        self.lookback = lookback
        self.candle_history = []

    def update(self, candle: Candle):
        self.candle_history.append(candle)
        if len(self.candle_history) > self.lookback:
            self.candle_history.pop(0)

    def compute(self) -> float:
        if not self.candle_history:
            return 0.0
        signals = []
        for c in self.candle_history:
            rng = c.high - c.low
            signals.append(0.0 if rng == 0 else (c.close - c.open) / rng)
        weights = np.exp(np.linspace(-1, 0, len(signals)))
        weights /= weights.sum()
        return float(np.dot(weights, signals))

    def signal_strength(self) -> str:
        s = self.compute()
        if s >  0.5: return "🔴 Strong Bull → Expect Reversal DOWN"
        if s >  0.2: return "🟠 Mild Bull → Slight Reversal"
        if s < -0.5: return "🟢 Strong Bear → Expect Reversal UP"
        if s < -0.2: return "🟡 Mild Bear → Slight Reversal"
        return "⚪ Neutral → No Signal"

# ═══════════════════════════════════════════════════════════
# 3. QUOTE CALCULATOR
# ═══════════════════════════════════════════════════════════

class QuoteCalculator:
    def __init__(self, gamma=0.1, alpha_weight=0.3,
                 spread_vol_multiplier=4.0, order_size=0.01):
        self.gamma            = gamma
        self.alpha_weight     = alpha_weight
        self.spread_vol_mult  = spread_vol_multiplier
        self.order_size       = order_size

    def compute_volatility(self, price_history: list) -> float:
        if len(price_history) < 2:
            return 0.001
        returns = np.diff(np.log(price_history[-20:]))
        return float(np.std(returns)) if len(returns) > 0 else 0.001

    def get_quotes(self, mid, inventory, bar_portion,
                   price_history, T_remaining=1.0) -> Quote:
        sigma  = self.compute_volatility(price_history)
        spread = max(self.spread_vol_mult * sigma * mid, mid * 0.001)

        inventory_skew    = inventory * self.gamma * sigma**2 * T_remaining * mid
        signal_skew       = -bar_portion * self.alpha_weight * sigma * mid
        reservation_price = mid - inventory_skew + signal_skew

        return Quote(
            bid_price = round(reservation_price - spread / 2, 4),
            ask_price = round(reservation_price + spread / 2, 4),
            bid_size  = self.order_size,
            ask_size  = self.order_size,
            signal    = bar_portion,
            spread    = round(spread, 6),
        )

# ═══════════════════════════════════════════════════════════
# 4. ORDER MANAGER (REST)
# ═══════════════════════════════════════════════════════════

class OrderManager:
    def __init__(self, exchange_name="binance", demo_mode=True):
        self.demo_mode     = demo_mode
        self.active_orders = {}
        self.order_counter = 0
        if not demo_mode:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange  = exchange_class({
                "apiKey":  st.secrets["api_key"],
                "secret":  st.secrets["api_secret"],
                "enableRateLimit": True,
            })

    def place_order(self, side, price, size) -> Order:
        self.order_counter += 1
        order_id = f"demo_{self.order_counter}"
        order = Order(order_id=order_id, side=side,
                      price=price, size=size, status="open")
        self.active_orders[order_id] = order
        print(f"[ORDER] Placed {side.upper()} @ ${price:.4f} size={size}")
        return order

    def cancel_order(self, order_id):
        if order_id in self.active_orders:
            del self.active_orders[order_id]

    def cancel_all(self):
        for order_id in list(self.active_orders.keys()):
            self.cancel_order(order_id)

    def simulate_fills(self, current_price, inventory):
        pnl_delta = 0.0
        for order_id, order in list(self.active_orders.items()):
            if order.side == "buy" and current_price <= order.price:
                inventory += order.size
                pnl_delta -= order.price * order.size
                del self.active_orders[order_id]
                print(f"[FILL] BUY filled @ ${order.price:.4f}")
            elif order.side == "sell" and current_price >= order.price:
                inventory -= order.size
                pnl_delta += order.price * order.size
                del self.active_orders[order_id]
                print(f"[FILL] SELL filled @ ${order.price:.4f}")
        return inventory, pnl_delta

# ═══════════════════════════════════════════════════════════
# 5. WEBSOCKET FEED — writes to SHARED_STATE (NOT session_state)
#    Note: shared_state is now passed in explicitly rather than
#    relying on a module global, so the thread is guaranteed to
#    keep writing into the same dict for its entire lifetime
#    even if something odd happens to the module namespace.
# ═══════════════════════════════════════════════════════════

async def stream_market_data(symbol: str, exchange_name: str, shared_state: dict):
    exchange = getattr(ccxtpro, exchange_name)()
    print(f"[WS] Connecting to {exchange_name} for {symbol}...")

    try:
        while shared_state["running"]:
            try:
                # Watch orderbook and ticker together
                orderbook, ticker = await asyncio.gather(
                    exchange.watch_order_book(symbol, limit=5),
                    exchange.watch_ticker(symbol),
                )

                # ── Write to SHARED_STATE (thread-safe plain dict) ──
                if orderbook["bids"] and orderbook["asks"]:
                    shared_state["best_bid"]  = orderbook["bids"][0][0]
                    shared_state["best_ask"]  = orderbook["asks"][0][0]
                    shared_state["mid_price"] = (
                        shared_state["best_bid"] + shared_state["best_ask"]
                    ) / 2
                    shared_state["orderbook"] = {
                        "bids": orderbook["bids"][:5],
                        "asks": orderbook["asks"][:5],
                    }
                    print(f"[WS] bid={shared_state['best_bid']:.4f} "
                          f"ask={shared_state['best_ask']:.4f} "
                          f"mid={shared_state['mid_price']:.4f}")

                shared_state["last_price"] = ticker["last"]

                # Price history for volatility
                shared_state["price_history"].append(ticker["last"])
                if len(shared_state["price_history"]) > 200:
                    shared_state["price_history"].pop(0)

                # Build 1-min candle from 60 ticks
                shared_state["tick_buffer"].append(ticker["last"])
                if len(shared_state["tick_buffer"]) >= 60:
                    prices = shared_state["tick_buffer"]
                    shared_state["latest_candle"] = Candle(
                        open=prices[0], high=max(prices),
                        low=min(prices),  close=prices[-1],
                        volume=ticker.get("baseVolume", 0),
                    )
                    shared_state["tick_buffer"] = []
                    print("[WS] New candle built")

            except Exception as e:
                print(f"[WS] Stream error: {e}, retrying in 3s...")
                await asyncio.sleep(3)

    finally:
        await exchange.close()
        print("[WS] Connection closed")


def start_websocket_thread(symbol: str, exchange_name: str, shared_state: dict):
    # Guard: only start once per browser session
    if st.session_state.get("ws_started"):
        return

    st.session_state.ws_started = True
    shared_state["running"]     = True

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                stream_market_data(symbol, exchange_name, shared_state)
            )
        except Exception as e:
            print(f"[WS] Thread crashed: {e}")
            shared_state["running"] = False

    threading.Thread(target=run, daemon=True).start()
    print("[WS] Thread launched")

# ═══════════════════════════════════════════════════════════
# 6. STRATEGY LOOP — also reads/writes SHARED_STATE
# ═══════════════════════════════════════════════════════════

def run_strategy_loop(signal_engine, quote_calc,
                      order_manager, shared_state: dict, refresh_seconds=60):
    print("[STRATEGY] Loop started")

    while shared_state["running"]:
        mid = shared_state.get("mid_price")
        if mid is None:
            time.sleep(1)
            continue

        # Update signal
        candle = shared_state.get("latest_candle")
        if candle:
            signal_engine.update(candle)

        bar_portion   = signal_engine.compute()
        price_history = shared_state.get("price_history", [mid])
        inventory     = shared_state.get("inventory", 0.0)

        # Compute quotes
        quote = quote_calc.get_quotes(
            mid=mid, inventory=inventory,
            bar_portion=bar_portion, price_history=price_history,
        )

        # Cancel old, place new
        order_manager.cancel_all()
        order_manager.place_order("buy",  quote.bid_price, quote.bid_size)
        order_manager.place_order("sell", quote.ask_price, quote.ask_size)

        # Simulate fills
        if order_manager.demo_mode:
            new_inv, pnl_delta = order_manager.simulate_fills(mid, inventory)
            shared_state["inventory"] = new_inv
            shared_state["pnl"]      = shared_state.get("pnl", 0.0) + pnl_delta

        # Push to SHARED_STATE for UI
        shared_state["current_quote"] = quote
        shared_state["signal_str"]    = signal_engine.signal_strength()
        shared_state["bar_portion"]   = bar_portion
        shared_state["active_orders"] = list(order_manager.active_orders.values())

        history = shared_state.get("quote_history", [])
        history.append({
            "time":      time.time(),
            "mid":       mid,
            "bid":       quote.bid_price,
            "ask":       quote.ask_price,
            "signal":    bar_portion,
            "inventory": inventory,
        })
        if len(history) > 100:
            history.pop(0)
        shared_state["quote_history"] = history

        print(f"[STRATEGY] bid={quote.bid_price:.4f} "
              f"ask={quote.ask_price:.4f} signal={bar_portion:.4f}")

        time.sleep(refresh_seconds)

# ═══════════════════════════════════════════════════════════
# 7. STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "mm_running": False,
        "ws_started": False,
        "symbol":     "BTC/USDT",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_dashboard():
    st.title("📊 PMM Bar Portion Market Maker")
    st.caption("Based on: Market Making in Crypto — Stoikov et al., Cornell 2024")

    # ── SIDEBAR ────────────────────────────────────────────
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
        order_size   = st.number_input("Order Size", value=0.01, step=0.001)

        st.divider()
        demo_mode = st.toggle("Demo Mode (no real orders)", value=True)

        col_start, col_stop = st.columns(2)
        start_btn = col_start.button("▶ Start", type="primary",
                                      disabled=st.session_state.mm_running)
        stop_btn  = col_stop.button("⏹ Stop",
                                     disabled=not st.session_state.mm_running)

    # ── START ──────────────────────────────────────────────
    if start_btn:
        st.session_state.mm_running = True
        st.session_state.symbol     = symbol

        signal_engine = BarPortionSignal(lookback=5)
        quote_calc    = QuoteCalculator(
            gamma=gamma, alpha_weight=alpha_weight,
            spread_vol_multiplier=spread_mult, order_size=order_size,
        )
        order_manager = OrderManager(exchange_name, demo_mode=demo_mode)

        start_websocket_thread(symbol, exchange_name, SHARED_STATE)

        threading.Thread(
            target=run_strategy_loop,
            args=(signal_engine, quote_calc, order_manager, SHARED_STATE, refresh_secs),
            daemon=True,
        ).start()

        st.success("✅ Market maker started! Watch terminal for live data.")

    # ── STOP ───────────────────────────────────────────────
    if stop_btn:
        st.session_state.mm_running = False
        st.session_state.ws_started = False
        SHARED_STATE["running"]     = False
        st.warning("⏹ Market maker stopped.")

    # Auto-refresh UI every 1 second
    st.autorefresh = st.empty()  # placeholder to avoid unused-import warnings if removed
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="ui_refresh")

    # ── DEBUG EXPANDER (remove after confirmed working) ────
    with st.expander("🔍 Debug — Raw SHARED_STATE", expanded=False):
        st.write("running:",     SHARED_STATE["running"])
        st.write("last_price:",  SHARED_STATE["last_price"])
        st.write("best_bid:",    SHARED_STATE["best_bid"])
        st.write("best_ask:",    SHARED_STATE["best_ask"])
        st.write("mid_price:",   SHARED_STATE["mid_price"])
        st.write("price_history count:", len(SHARED_STATE["price_history"]))
        st.write("orderbook bids:", SHARED_STATE["orderbook"]["bids"][:2])
        st.write("orderbook asks:", SHARED_STATE["orderbook"]["asks"][:2])

    st.divider()

    # ── ROW 1: LIVE METRICS ────────────────────────────────
    st.subheader("📡 Live Market Data (WebSocket)")
    m1, m2, m3, m4, m5 = st.columns(5)

    last_price = SHARED_STATE["last_price"]
    best_bid   = SHARED_STATE["best_bid"]
    best_ask   = SHARED_STATE["best_ask"]
    inventory  = SHARED_STATE["inventory"]
    pnl        = SHARED_STATE["pnl"]

    m1.metric("Last Price", f"${last_price:,.4f}" if last_price else "⏳ connecting...")
    m2.metric("Best Bid",   f"${best_bid:,.4f}"   if best_bid   else "⏳ connecting...")
    m3.metric("Best Ask",   f"${best_ask:,.4f}"   if best_ask   else "⏳ connecting...")
    m4.metric("Inventory",  f"{inventory:.4f}")
    m5.metric("PnL",        f"${pnl:.4f}", delta=f"${pnl:.4f}")

    st.divider()

    # ── ROW 2: SIGNAL + QUOTES + ORDERS ───────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 Bar Portion Signal")
        bp = SHARED_STATE["bar_portion"]
        st.metric("Bar Portion Value", f"{bp:.4f}")
        st.info(SHARED_STATE["signal_str"])
        st.progress(
            float(np.clip((bp + 1) / 2, 0, 1)),
            text=f"Signal: {bp:.3f}  (−1=Bear | 0=Neutral | +1=Bull)"
        )

    with col2:
        st.subheader("💬 Our Quotes")
        quote = SHARED_STATE["current_quote"]
        if quote:
            q1, q2 = st.columns(2)
            q1.metric("BID",  f"${quote.bid_price:,.4f}", delta="we buy here")
            q2.metric("ASK",  f"${quote.ask_price:,.4f}", delta="we sell here")
            st.metric("Spread", f"${quote.spread:.6f}")
            mid = SHARED_STATE["mid_price"]
            if mid:
                st.caption(f"Mid: ${mid:,.4f} | "
                           f"Bid offset: ${mid - quote.bid_price:.4f} | "
                           f"Ask offset: ${quote.ask_price - mid:.4f}")
        else:
            st.info("⏳ Waiting for first quote cycle...")

    with col3:
        st.subheader("📋 Active Orders")
        orders = SHARED_STATE["active_orders"]
        if orders:
            st.dataframe(pd.DataFrame([{
                "ID":    o.order_id[-6:],
                "Side":  o.side.upper(),
                "Price": f"${o.price:,.4f}",
                "Size":  o.size,
                "Status": o.status,
            } for o in orders]), use_container_width=True, hide_index=True)
        else:
            st.info("No active orders")

    st.divider()

    # ── ROW 3: ORDER BOOK + QUOTE CHART ───────────────────
    col_ob, col_chart = st.columns([1, 2])

    with col_ob:
        st.subheader("📖 Live Order Book")
        ob = SHARED_STATE["orderbook"]
        if ob["bids"] and ob["asks"]:
            asks_df = pd.DataFrame(ob["asks"][:5], columns=["Price", "Size"])
            bids_df = pd.DataFrame(ob["bids"][:5], columns=["Price", "Size"])
            asks_df["Price"] = asks_df["Price"].apply(lambda x: f"${x:,.4f}")
            bids_df["Price"] = bids_df["Price"].apply(lambda x: f"${x:,.4f}")
            asks_df["Side"]  = "🔴 ASK"
            bids_df["Side"]  = "🟢 BID"
            st.dataframe(asks_df.iloc[::-1], use_container_width=True, hide_index=True)
            st.markdown("<center>─── spread ───</center>", unsafe_allow_html=True)
            st.dataframe(bids_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⏳ Waiting for order book...")

    with col_chart:
        st.subheader("📉 Quote History (mid / bid / ask)")
        history = SHARED_STATE["quote_history"]
        if len(history) > 2:
            hist_df = pd.DataFrame(history)
            hist_df["time"] = pd.to_datetime(hist_df["time"], unit="s")
            st.line_chart(hist_df.set_index("time")[["mid", "bid", "ask"]])
        else:
            st.info("⏳ Collecting data — quotes appear every refresh cycle...")

    st.divider()

    # ── ROW 4: SIGNAL + INVENTORY OVER TIME ───────────────
    col_sig, col_inv = st.columns(2)

    with col_sig:
        st.subheader("📊 Signal Over Time")
        if len(history) > 2:
            sig_df = pd.DataFrame(history)[["time", "signal"]]
            sig_df["time"] = pd.to_datetime(sig_df["time"], unit="s")
            st.line_chart(sig_df.set_index("time"))

    with col_inv:
        st.subheader("📦 Inventory Over Time")
        if len(history) > 2:
            inv_df = pd.DataFrame(history)[["time", "inventory"]]
            inv_df["time"] = pd.to_datetime(inv_df["time"], unit="s")
            st.line_chart(inv_df.set_index("time"))


# ═══════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════
init_session_state()
render_dashboard()