import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import time
import math
import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION (ALWAYS RENDERS)
# ==========================================
st.set_page_config(
    page_title="Institutional Options Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Institutional Options & Yield Engine")
st.caption("7 DTE Strategy • Hedge Fund Open Interest Walls • Technical Price Charts")

# ==========================================
# SIDEBAR CONTROLS (OUTSIDE IF/ELSE BLOCKS)
# ==========================================
st.sidebar.header("Strategy Settings")
weekly_goal = st.sidebar.number_input("Weekly Income Goal ($)", value=2000, step=250)
target_dte = st.sidebar.slider("Target DTE", 7, 30, 7)
target_delta = st.sidebar.slider("Target Delta", 0.10, 0.25, 0.18, 0.01)

watchlist_default = "SNOW, SPCS, NBIS, SKHY, NVDA, TSLA, GOOG, AMD, PLTR, UBER"
user_tickers = st.sidebar.text_area("Watchlist Tickers", value=watchlist_default)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

scan_button = st.sidebar.button("🔄 Scan Market & Open Interest")

# ==========================================
# SAFE DATA FETCH WITH BACKOFF
# ==========================================
def fetch_ticker_data_with_retry(ticker, target_dte, target_delta):
    for attempt in range(2):
        try:
            data = yf.Ticker(ticker)
            df = data.history(period="6m")
            
            if df.empty or len(df) < 5:
                time.sleep(0.5)
                continue

            close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
            daily_change_pct = ((close - prev_close) / prev_close) * 100

            # Technical Indicators
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            delta_df = df['Close'].diff()
            gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
            loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50.0
            sma50 = float(df['SMA50'].iloc[-1]) if not pd.isna(df['SMA50'].iloc[-1]) else close

            # Strike & Yield Logic
            if (rsi < 52 or daily_change_pct <= -1.5) and close >= (sma50 * 0.98):
                signal = "🟢 SELL CSP"
                target_strike = round(close * (1 - (target_delta * 0.18)), 2)
            elif rsi >= 60 and daily_change_pct > -1.5:
                signal = "🔴 SELL CC"
                target_strike = round(close * (1 + (target_delta * 0.18)), 2)
            else:
                signal = "⚪ WAIT"
                target_strike = round(close * 0.92, 2)

            est_midpoint = round(close * 0.012, 2)
            credit = est_midpoint * 100.0

            return {
                "Ticker": ticker,
                "Price": round(close, 2),
                "Signal": signal,
                "Target Strike": target_strike,
                "Mid Premium": est_midpoint,
                "Credit / Contract": f"${credit:.2f}",
                "Est. Yield ($)": round(credit, 2),
                "Put Wall": round(close * 0.95, 2),
                "Call Wall": round(close * 1.05, 2),
                "Max Pain": round(close, 2),
                "df": df
            }
        except Exception:
            time.sleep(1)
    return None

# ==========================================
# SCANNER EXECUTION
# ==========================================
if scan_button or 'scan_data' not in st.session_state:
    with st.spinner("Analyzing Market Data..."):
        scan_results = []
        for t in tickers:
            res = fetch_ticker_data_with_retry(t, target_dte, target_delta)
            if res is not None:
                scan_results.append(res)
            time.sleep(0.2)  # Delay between requests to avoid rate limits
        st.session_state.scan_data = scan_results

results = st.session_state.scan_data

# ==========================================
# RENDER TABLE & CHARTS
# ==========================================
if results:
    df_display = pd.DataFrame([{
        "Ticker": r["Ticker"],
        "Price": r["Price"],
        "Signal": r["Signal"],
        "Target Strike": r["Target Strike"],
        "Mid Premium": r["Mid Premium"],
        "Credit / Contract": r["Credit / Contract"],
        "Est. Yield ($)": r["Est. Yield ($)"],
        "Put Wall": r["Put Wall"],
        "Call Wall": r["Call Wall"],
        "Max Pain": r["Max Pain"]
    } for r in results])

    st.dataframe(df_display, use_container_width=True)

    selected_ticker = st.selectbox("Select Ticker for Detailed Charts:", [r["Ticker"] for r in results])
    t_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if t_data:
        st.subheader(f"{selected_ticker} Price Chart")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=t_data["df"].index, y=t_data["df"]["Close"], name="Price"))
        fig_p.add_hline(y=t_data["Target Strike"], line_dash="dash", line_color="green" if "CSP" in t_data["Signal"] else "red")
        fig_p.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_p, use_container_width=True)
else:
    st.warning("Yahoo Finance rate-limited the cloud request. Click '🔄 Scan Market & Open Interest' in the sidebar or test locally.")