import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Portfolio Yield Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2A2E39;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# BLACK-SCHOLES GREEK ENGINE (NO EXTERNAL DEPS)
# ==========================================
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0, 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    cdf_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    pdf_d1 = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1**2)
    
    if option_type.lower() == "call":
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1.0
        
    vega = S * pdf_d1 * math.sqrt(T) / 100.0
    return round(float(delta), 3), round(float(vega), 3)

# ==========================================
# SINGLE TICKER PROCESSOR
# ==========================================
def process_single_ticker(ticker, target_dte=21, target_delta=0.15):
    try:
        data = yf.Ticker(ticker)
        df = data.history(period="1y")
        
        if len(df) < 15:
            return None

        close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
        daily_change_pct = ((close - prev_close) / prev_close) * 100

        df['SMA50'] = df['Close'].rolling(window=50).mean()
        delta_df = df['Close'].diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])
        sma50 = float(df['SMA50'].iloc[-1]) if not pd.isna(df['SMA50'].iloc[-1]) else close

        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        volatility_est = float(log_returns.tail(21).std() * np.sqrt(252))
        if np.isnan(volatility_est) or volatility_est == 0:
            volatility_est = 0.25

        dte_multiplier = 2.1 if target_dte <= 21 else 3.2

        # ==========================================
        # SMART SIGNAL ENGINE (WITH DAILY DROP GUARD)
        # ==========================================
        if (rsi < 52 or daily_change_pct <= -2.0) and close >= (sma50 * 0.98):
            signal = "🟢 SELL CSP"
            opt_type = "put"
            est_strike = close * (1 - (target_delta * volatility_est * (target_dte / 30.0)))
            est_premium = close * target_delta * (volatility_est * dte_multiplier)

        elif rsi >= 60 and daily_change_pct > -1.5:
            signal = "🔴 SELL CC"
            opt_type = "call"
            est_strike = close * (1 + (target_delta * volatility_est * (target_dte / 30.0)))
            est_premium = close * target_delta * (volatility_est * dte_multiplier)

        else:
            signal = "⚪ WAIT"
            opt_type = "put"
            est_strike = close * (1 - (target_delta * volatility_est * (target_dte / 30.0)))
            est_premium = close * target_delta * (volatility_est * dte_multiplier)

        T_years = target_dte / 365.0
        delta_val, vega_val = calculate_greeks(
            S=close, K=est_strike, T=T_years, r=0.045, sigma=volatility_est, option_type=opt_type
        )

        return {
            "Ticker": ticker,
            "Price": round(close, 2),
            "Daily Change %": round(daily_change_pct, 2),
            "RSI (14)": round(rsi, 1),
            "50-SMA": round(sma50, 2),
            "Signal": signal,
            "Target Strike": round(est_strike, 2),
            "Est. Premium": round(est_premium, 2),
            "Est. Delta": delta_val,
            "Est. Vega": vega_val,
            "Ann. Volatility": f"{round(volatility_est * 100, 1)}%"
        }
    except Exception as e:
        return None

# ==========================================
# PARALLEL DATA FETCHING
# ==========================================
def fetch_all_tickers(ticker_list, target_dte, target_delta):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_single_ticker, ticker, target_dte, target_delta) 
            for ticker in ticker_list
        ]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)
    return pd.DataFrame(results)

# Alias for backward compatibility with cached function calls
fetch_and_analyze_data = fetch_all_tickers

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.title("🎯 Options Yield & Portfolio Scanner")
st.caption("14–21 DTE Engine with Daily Drop Guards & Black-Scholes Greeks")

st.sidebar.header("Strategy Controls")
portfolio_size = st.sidebar.number_input("Portfolio Capital ($)", value=600000, step=25000)
target_dte = st.sidebar.slider("Days to Expiration (DTE)", min_value=7, max_value=45, value=21, step=1)
target_delta = st.sidebar.slider("Target Delta", min_value=0.10, max_value=0.30, value=0.15, step=0.01)

default_watchlist = "GOOG, TMUS, SKHY, NVDA, TSLA, AMD, SPY, QQQ"
user_tickers = st.sidebar.text_area("Watchlist (Comma Separated)", value=default_watchlist)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

if st.sidebar.button("🔄 Refresh Market Scanner") or 'scan_df' not in st.session_state:
    with st.spinner("Processing market data & calculating Greeks..."):
        st.session_state.scan_df = fetch_all_tickers(tickers, target_dte, target_delta)

df_results = st.session_state.scan_df

if not df_results.empty:
    csp_count = len(df_results[df_results["Signal"] == "🟢 SELL CSP"])
    cc_count = len(df_results[df_results["Signal"] == "🔴 SELL CC"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Watchlist Count", len(df_results))
    col2.metric("CSP Opportunities", f"{csp_count} Tickers")
    col3.metric("CC Opportunities", f"{cc_count} Tickers")
    col4.metric("Max Position Limit (20%)", f"${portfolio_size * 0.20:,.0f}")

    st.markdown("---")
    st.subheader("Live Scanner Results")
    
    def highlight_signals(val):
        if val == "🟢 SELL CSP":
            return "background-color: #113824; color: #4EFE96;"
        elif val == "🔴 SELL CC":
            return "background-color: #4A151B; color: #FF6B6B;"
        return "color: #888888;"

    styled_df = df_results.style.map(highlight_signals, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("Position Execution Planner")
    selected_ticker = st.selectbox("Select Ticker to Execute", df_results["Ticker"].unique())
    
    ticker_row = df_results[df_results["Ticker"] == selected_ticker].iloc[0]
    
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Current Price", f"${ticker_row['Price']}")
    pcol2.metric("Target Strike Price", f"${ticker_row['Target Strike']}")
    pcol3.metric("Est. Upfront Credit", f"${ticker_row['Est. Premium'] * 100:,.2f} / contract")

    st.info(f"**Execution Note:** Selected DTE is set to **{target_dte} Days**. "
            f"Close contract automatically upon reaching **50% max profit** or at **21 DTE**.")
else:
    st.warning("No valid ticker data returned. Please verify your watchlist symbols.")
