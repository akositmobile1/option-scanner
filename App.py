import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Institutional Options Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Institutional Options & Yield Engine")
st.caption("7 DTE Strategy • Fast-Info Yahoo Feed • Monday Technical & Volatility Overlays")

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
# LIGHTWEIGHT YFINANCE FETCH & TECHNICALS
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

    # Historical Data for Indicators
    try:
        t_obj = yf.Ticker(ticker)
        df_hist = t_obj.history(period="6m")
        if isinstance(df_hist.columns, pd.MultiIndex):
            df_hist = df_hist.xs(ticker, level=1, axis=1)

        if not df_hist.empty and len(df_hist) >= 30:
            close_s = df_hist['Close']
            
            # 1. EMAs
            df_hist['EMA20'] = close_s.ewm(span=20, adjust=False).mean()
            df_hist['EMA50'] = close_s.ewm(span=50, adjust=False).mean()

            # 2. RSI (14)
            delta_df = close_s.diff()
            gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
            loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_hist['RSI'] = 100 - (100 / (1 + rs))
            current_rsi = float(df_hist['RSI'].iloc[-1]) if not pd.isna(df_hist['RSI'].iloc[-1]) else 50.0

            # 3. ATR (14) & Expected 5-Day Move
            high_low = df_hist['High'] - df_hist['Low']
            high_close = np.abs(df_hist['High'] - close_s.shift())
            low_close = np.abs(df_hist['Low'] - close_s.shift())
            tr = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
            df_hist['ATR'] = tr.rolling(14).mean()
            current_atr = float(df_hist['ATR'].iloc[-1]) if not pd.isna(df_hist['ATR'].iloc[-1]) else close * 0.02
            
            weekly_move = current_atr * np.sqrt(5)
            upper_atr_bound = close + (1.2 * weekly_move)
            lower_atr_bound = close - (1.2 * weekly_move)
        else:
            df_hist = None
            current_rsi = 50.0
            lower_atr_bound = close * 0.92
            upper_atr_bound = close * 1.08
    except Exception:
        df_hist = None
        current_rsi = 50.0
        lower_atr_bound = close * 0.92
        upper_atr_bound = close * 1.08

    # Rules Engine Signal Logic
    if daily_change_pct <= -1.0 or current_rsi < 45:
        signal = "🟢 SELL CSP"
        target_strike = round(close * (1 - (target_delta * 0.18)), 2)
    elif daily_change_pct >= 1.5 or current_rsi > 65:
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
        "Max Pain": round(close, 2),
        "RSI": round(current_rsi, 1),
        "Lower ATR": round(lower_atr_bound, 2),
        "Upper ATR": round(upper_atr_bound, 2),
        "df_hist": df_hist
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
    st.dataframe(styled_df, use_container_width=True, height=360)

    # ==========================================
    # ADDED: MONDAY TECHNICAL ANALYSIS CHART
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Technical Confirmation & Volatility Bounds")

    selected_ticker = st.selectbox("Select Ticker for Detailed Setup Verification:", [r["Ticker"] for r in results])
    t_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if t_data and t_data["df_hist"] is not None:
        df_chart = t_data["df_hist"]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                f"{selected_ticker} Candlesticks, EMAs & Weekly ATR Bounds",
                "Volume",
                f"RSI 14 ({t_data['RSI']})"
            ),
            row_heights=[0.6, 0.2, 0.2]
        )

        # Panel 1: Price, EMAs, and Target Strike / ATR Bounds
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'], close=df_chart['Close'],
            name="Price"
        ), row=1, col=1)

        if 'EMA20' in df_chart.columns:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], mode='lines', name='20 EMA', line=dict(color='#00F0FF', width=1.5)), row=1, col=1)
        if 'EMA50' in df_chart.columns:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA50'], mode='lines', name='50 EMA', line=dict(color='#FFD166', width=1.5)), row=1, col=1)

        # Target Strike Line
        strike_color = "#4EFE96" if "CSP" in t_data["Signal"] else "#FF6B6B"
        fig.add_hline(
            y=t_data["Target Strike"], line_dash="dash", line_color=strike_color, line_width=2,
            annotation_text=f"Target Strike: ${t_data['Target Strike']:.2f}", annotation_position="top right", row=1, col=1
        )

        # Weekly ATR Guard Rails
        fig.add_hline(
            y=t_data["Lower ATR"], line_dash="dot", line_color="#22C55E", opacity=0.7,
            annotation_text=f"ATR Support: ${t_data['Lower ATR']:.2f}", annotation_position="bottom left", row=1, col=1
        )
        fig.add_hline(
            y=t_data["Upper ATR"], line_dash="dot", line_color="#EF4444", opacity=0.7,
            annotation_text=f"ATR Resist: ${t_data['Upper ATR']:.2f}", annotation_position="top left", row=1, col=1
        )

        # Panel 2: Volume
        vol_colors = ['#EF4444' if row['Open'] > row['Close'] else '#22C55E' for _, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name="Volume", marker_color=vol_colors), row=2, col=1)

        # Panel 3: RSI
        if 'RSI' in df_chart.columns:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], mode='lines', name='RSI', line=dict(color='#A855F7', width=1.5)), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#22C55E", row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=680,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart history currently unavailable for this ticker.")
else:
    st.info("👈 Click '🔄 Scan Market Data' to trigger fresh Yahoo quote updates.")
