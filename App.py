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
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Institutional Options Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Institutional Options & Monday Setup Engine")
st.caption("7 DTE Strategy • Weekly ATR Expected Move • Technical Confirmation Overlays")

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

scan_button = st.sidebar.button("🔄 Scan Market & Calculate Indicators", use_container_width=True)

# ==========================================
# TECHNICAL ANALYSIS & DATA FETCHING
# ==========================================
def process_ticker(symbol, target_dte, target_delta):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="6m")
        
        if df.empty or len(df) < 50:
            return None

        # Standardize DataFrame columns
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=1, axis=1)

        close_series = df['Close']
        current_price = float(close_series.iloc[-1])
        prev_close = float(close_series.iloc[-2]) if len(close_series) >= 2 else current_price
        daily_change_pct = ((current_price - prev_close) / prev_close) * 100

        # Technical Indicators Calculations
        df['EMA20'] = close_series.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close_series.ewm(span=50, adjust=False).mean()

        # Relative Strength Index (RSI - 14)
        delta_df = close_series.diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        current_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0

        # Average True Range (ATR - 14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - close_series.shift())
        low_close = np.abs(df['Low'] - close_series.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        current_atr = float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02

        # Expected 5-Day Move Formula (ATR * sqrt(5))
        weekly_expected_move = current_atr * np.sqrt(5)
        upper_atr_bound = current_price + (1.5 * weekly_expected_move)
        lower_atr_bound = current_price - (1.5 * weekly_expected_move)

        # Rules Engine Signal Logic
        ema50_val = float(df['EMA50'].iloc[-1])
        if (current_rsi < 48 or daily_change_pct <= -1.0) and current_price >= (ema50_val * 0.96):
            signal = "🟢 SELL CSP"
            target_strike = round(current_price * (1 - (target_delta * 0.18)), 2)
        elif current_rsi >= 62 and daily_change_pct >= 1.2:
            signal = "🔴 SELL CC"
            target_strike = round(current_price * (1 + (target_delta * 0.18)), 2)
        else:
            signal = "⚪ WAIT"
            target_strike = round(current_price * 0.95, 2)

        est_midpoint = round(current_price * 0.012, 2)
        credit_per_contract = est_midpoint * 100.0

        return {
            "Ticker": symbol,
            "Price": round(current_price, 2),
            "Change %": round(daily_change_pct, 2),
            "RSI": round(current_rsi, 1),
            "Signal": signal,
            "Target Strike": target_strike,
            "Mid Premium": est_midpoint,
            "Credit / Contract": f"${credit_per_contract:.2f}",
            "Est. Yield ($)": round(credit_per_contract, 2),
            "Weekly Upper ATR": round(upper_atr_bound, 2),
            "Weekly Lower ATR": round(lower_atr_bound, 2),
            "df": df
        }
    except Exception:
        return None

# ==========================================
# SCANNER EXECUTION
# ==========================================
if scan_button or 'scan_data' not in st.session_state:
    with st.spinner("Analyzing Candlesticks, EMAs, RSI & ATR Bands..."):
        results = []
        failed_tickers = []
        
        for t in tickers:
            res = process_ticker(t, target_dte, target_delta)
            if res:
                results.append(res)
            else:
                failed_tickers.append(t)
            time.sleep(0.08)
            
        st.session_state.scan_data = results
        st.session_state.failed_tickers = failed_tickers

results = st.session_state.get('scan_data', [])
failed = st.session_state.get('failed_tickers', [])

# ==========================================
# RENDER TABLE
# ==========================================
if failed:
    st.warning(f"⚠️ Could not process market history for: {', '.join(failed)}.")

if results:
    df_display = pd.DataFrame([{
        "Ticker": r["Ticker"],
        "Price": f"${r['Price']:.2f}",
        "Change %": f"{r['Change %']:+.2f}%",
        "RSI (14)": r["RSI"],
        "Signal": r["Signal"],
        "Target Strike": f"${r['Target Strike']:.2f}",
        "Mid Premium": f"${r['Mid Premium']:.2f}",
        "Credit / Contract": r["Credit / Contract"],
        "Lower ATR Guard": f"${r['Weekly Lower ATR']:.2f}",
        "Upper ATR Guard": f"${r['Weekly Upper ATR']:.2f}"
    } for r in results])

    st.subheader("📋 Monday Market Signals & Volatility Guards")
    
    def highlight_signal(val):
        if "SELL CSP" in str(val):
            return "background-color: #113824; color: #4EFE96; font-weight: bold;"
        elif "SELL CC" in str(val):
            return "background-color: #4A151B; color: #FF6B6B; font-weight: bold;"
        return "color: #888888;"

    styled_df = df_display.style.map(highlight_signal, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=360)

    # ==========================================
    # MONDAY SETUP CHART ENGINE
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Technical Setup & Volatility Bounds")

    selected_ticker = st.selectbox("Select Ticker for Setup Verification:", [r["Ticker"] for r in results])
    t_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if t_data:
        df_chart = t_data["df"]
        
        # 3-Panel Subplot: Price/EMAs/Bands, Volume, RSI
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.04, 
            subplot_titles=(f"{selected_ticker} Candlesticks, Moving Averages & ATR Bounds", "Volume", "RSI (14) Indicator"),
            row_heights=[0.6, 0.2, 0.2]
        )

        # Panel 1: Candlesticks & Indicators
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'], close=df_chart['Close'],
            name="Price"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], mode='lines', name='20 EMA', line=dict(color='#00F0FF', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA50'], mode='lines', name='50 EMA', line=dict(color='#FFD166', width=1.5)), row=1, col=1)

        # Target Strike Highlight Line
        strike_color = "#4EFE96" if "CSP" in t_data["Signal"] else "#FF6B6B"
        fig.add_hline(y=t_data["Target Strike"], line_dash="dash", line_color=strike_color, line_width=2,
                      annotation_text=f"Target Strike: ${t_data['Target Strike']:.2f}", annotation_position="top right", row=1, col=1)

        # ATR Expected Move Bounds
        fig.add_hline(y=t_data["Weekly Lower ATR"], line_dash="dot", line_color="#22C55E", opacity=0.7,
                      annotation_text=f"Weekly ATR Support: ${t_data['Weekly Lower ATR']:.2f}", annotation_position="bottom left", row=1, col=1)
        fig.add_hline(y=t_data["Weekly Upper ATR"], line_dash="dot", line_color="#EF4444", opacity=0.7,
                      annotation_text=f"Weekly ATR Resist: ${t_data['Weekly Upper ATR']:.2f}", annotation_position="top left", row=1, col=1)

        # Panel 2: Volume
        colors = ['#EF4444' if row['Open'] > row['Close'] else '#22C55E' for _, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name="Volume", marker_color=colors), row=2, col=1)

        # Panel 3: RSI
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], mode='lines', name='RSI', line=dict(color='#A855F7', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22C55E", row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Press '🔄 Scan Market & Calculate Indicators' in the sidebar to load fresh technical data.")
