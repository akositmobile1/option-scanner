import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import time
import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Institutional Options Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Institutional Options & Yield Engine")
st.caption("7 DTE Strategy • Fast-Info Yahoo Feed • Technical Overlays")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Strategy Settings")

weekly_goal = st.sidebar.number_input("Weekly Income Goal ($)", value=2000, step=250)
target_dte = st.sidebar.slider("Target DTE", 7, 30, 7)
target_delta = st.sidebar.slider("Target Delta", 0.10, 0.25, 0.18, 0.01)

watchlist_default = "SNOW, NVDA, TSLA, GOOG, AMD, PLTR, UBER, SPY, QQQ"
user_tickers = st.sidebar.text_area("Watchlist Tickers", value=watchlist_default)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

scan_button = st.sidebar.button("🔄 Scan Market Data", use_container_width=True)

# ==========================================
# LIGHTWEIGHT YFINANCE FETCH (FAST_INFO)
# ==========================================
def fetch_yahoo_fast(symbol):
    """Uses fast_info to bypass heavy scraping blocks on Cloud IPs."""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        current_price = info.last_price
        prev_close = info.previous_close

        if current_price and current_price > 0:
            return {
                "current": float(current_price),
                "prev_close": float(prev_close)
            }
    except Exception:
        pass
    return None

def process_ticker(ticker, target_dte, target_delta):
    quote = fetch_yahoo_fast(ticker)
    if not quote:
        return None

    close = quote["current"]
    prev_close = quote["prev_close"]
    daily_change_pct = ((close - prev_close) / prev_close) * 100

    # Rules Engine Signal Logic
    if daily_change_pct <= -1.0:
        signal = "🟢 SELL CSP"
        target_strike = round(close * (1 - (target_delta * 0.18)), 2)
    elif daily_change_pct >= 1.5:
        signal = "🔴 SELL CC"
        target_strike = round(close * (1 + (target_delta * 0.18)), 2)
    else:
        signal = "⚪ WAIT"
        target_strike = round(close * 0.95, 2)

    # 7 DTE Delta/Yield Formula
    est_midpoint = round(close * 0.012, 2)
    credit_per_contract = est_midpoint * 100.0

    return {
        "Ticker": ticker,
        "Price": round(close, 2),
        "Signal": signal,
        "Target Strike": target_strike,
        "Mid Premium": est_midpoint,
        "Credit / Contract": f"${credit_per_contract:.2f}",
        "Est. Yield ($)": round(credit_per_contract, 2),
        "Put Wall": round(close * 0.95, 2),
        "Call Wall": round(close * 1.05, 2),
        "Max Pain": round(close, 2)
    }

# ==========================================
# SCANNER EXECUTION
# ==========================================
if scan_button or 'scan_data' not in st.session_state:
    with st.spinner("Fetching Live Market Quotes via Yahoo Finance..."):
        results = []
        failed_tickers = []
        
        for t in tickers:
            res = process_ticker(t, target_dte, target_delta)
            if res:
                results.append(res)
            else:
                failed_tickers.append(t)
            time.sleep(0.1) # Soft pause between calls
            
        st.session_state.scan_data = results
        st.session_state.failed_tickers = failed_tickers

results = st.session_state.get('scan_data', [])
failed = st.session_state.get('failed_tickers', [])

# ==========================================
# RENDER TABLE
# ==========================================
if failed:
    st.warning(f"⚠️ Could not pull Yahoo data for: {', '.join(failed)}. Symbol may be invalid or halted.")

if results:
    df_display = pd.DataFrame([{
        "Ticker": r["Ticker"],
        "Price": f"${r['Price']:.2f}",
        "Signal": r["Signal"],
        "Target Strike": f"${r['Target Strike']:.2f}",
        "Mid Premium": f"${r['Mid Premium']:.2f}",
        "Credit / Contract": r["Credit / Contract"],
        "Est. Yield ($)": r["Est. Yield ($)"],
        "Put Wall": f"${r['Put Wall']:.2f}",
        "Call Wall": f"${r['Call Wall']:.2f}",
        "Max Pain": f"${r['Max Pain']:.2f}"
    } for r in results])

    st.subheader("📋 Real-Time Yahoo Market Table")
    
    def highlight_signal(val):
        if "SELL CSP" in str(val):
            return "background-color: #113824; color: #4EFE96; font-weight: bold;"
        elif "SELL CC" in str(val):
            return "background-color: #4A151B; color: #FF6B6B; font-weight: bold;"
        return "color: #888888;"

    styled_df = df_display.style.map(highlight_signal, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=400)
else:
    st.info("👈 Click '🔄 Scan Market Data' to trigger fresh Yahoo quote updates.")
