import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Portfolio Yield Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BLACK-SCHOLES GREEK ENGINE
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
# LIVE OPTION CHAIN & BID-ASK MIDPOINT ENGINE
# ==========================================
def get_live_option_data(ticker_obj, close_price, opt_type, target_dte, target_delta):
    try:
        expirations = ticker_obj.expirations
        if not expirations:
            return None, None, None

        # Find expiration closest to target_dte
        target_days = target_dte
        best_exp = expirations[0]
        min_diff = 999
        
        for exp in expirations:
            # Calculate DTE roughly from expiration string
            exp_date = pd.to_datetime(exp)
            dte = (exp_date - pd.Timestamp.now()).days
            if abs(dte - target_days) < min_diff and dte > 0:
                min_diff = abs(dte - target_days)
                best_exp = exp

        chain = ticker_obj.option_chain(best_exp)
        options = chain.calls if opt_type == "call" else chain.puts

        if options.empty:
            return None, None, None

        # Calculate strike closest to target delta target
        if opt_type == "call":
            target_strike = close_price * (1 + (target_delta * 0.25 * (target_dte / 30.0)))
            # Filter OTM calls
            otm_opts = options[options['strike'] >= close_price]
        else:
            target_strike = close_price * (1 - (target_delta * 0.25 * (target_dte / 30.0)))
            # Filter OTM puts
            otm_opts = options[options['strike'] <= close_price]

        if otm_opts.empty:
            otm_opts = options

        # Find closest strike
        idx = (otm_opts['strike'] - target_strike).abs().idxmin()
        selected_option = otm_opts.loc[idx]

        strike = float(selected_option['strike'])
        bid = float(selected_option['bid'])
        ask = float(selected_option['ask'])

        # Calculate Midpoint
        if bid > 0 and ask > 0:
            midpoint = (bid + ask) / 2.0
        elif selected_option['lastPrice'] > 0:
            midpoint = float(selected_option['lastPrice'])
        else:
            midpoint = 1.0

        return strike, midpoint, best_exp
    except Exception:
        return None, None, None

# ==========================================
# TICKER PROCESSOR
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

        # Technical Indicators
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        delta_df = df['Close'].diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])
        sma50 = float(df['SMA50'].iloc[-1]) if not pd.isna(df['SMA50'].iloc[-1]) else close

        # Historical Volatility
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        volatility_est = float(log_returns.tail(21).std() * np.sqrt(252))
        if np.isnan(volatility_est) or volatility_est == 0:
            volatility_est = 0.25

        # Signal Engine with Daily Drop Guard
        if (rsi < 52 or daily_change_pct <= -2.0) and close >= (sma50 * 0.98):
            signal = "🟢 SELL CSP"
            opt_type = "put"
        elif rsi >= 60 and daily_change_pct > -1.5:
            signal = "🔴 SELL CC"
            opt_type = "call"
        else:
            signal = "⚪ WAIT"
            opt_type = "put"

        # Fetch Live Option Chain Midpoint & Strike
        live_strike, live_midpoint, best_exp = get_live_option_data(
            data, close, opt_type, target_dte, target_delta
        )

        # Fallback to estimated values if option chain is offline/unavailable
        if live_strike is None:
            if opt_type == "call":
                live_strike = close * (1 + (target_delta * volatility_est * (target_dte / 30.0)))
            else:
                live_strike = close * (1 - (target_delta * volatility_est * (target_dte / 30.0)))
            live_midpoint = close * target_delta * (volatility_est * 0.15)
            best_exp = f"{target_dte} DTE"

        # Black-Scholes Greeks
        T_years = target_dte / 365.0
        delta_val, vega_val = calculate_greeks(
            S=close, K=live_strike, T=T_years, r=0.045, sigma=volatility_est, option_type=opt_type
        )

        return {
            "Ticker": ticker,
            "Price": round(close, 2),
            "Daily Change %": round(daily_change_pct, 2),
            "RSI (14)": round(rsi, 1),
            "50-SMA": round(sma50, 2),
            "Signal": signal,
            "Target Strike": round(live_strike, 2),
            "Midpoint Premium ($/sh)": round(live_midpoint, 2),
            "Contract Credit ($)": round(live_midpoint * 100, 2),
            "Expiration": best_exp,
            "Est. Delta": delta_val,
            "Est. Vega": vega_val,
            "df_history": df
        }
    except Exception:
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
    return results

fetch_and_analyze_data = fetch_all_tickers

# ==========================================
# UI LAYOUT & DASHBOARD
# ==========================================
st.title("🎯 Options Yield & Portfolio Scanner")
st.caption("Live Midpoints, Technical Charts, Daily Drop Guards & Black-Scholes Greeks")

st.sidebar.header("Strategy Controls")
portfolio_size = st.sidebar.number_input("Portfolio Capital ($)", value=600000, step=25000)
target_dte = st.sidebar.slider("Days to Expiration (DTE)", min_value=7, max_value=45, value=21, step=1)
target_delta = st.sidebar.slider("Target Delta", min_value=0.10, max_value=0.30, value=0.15, step=0.01)

default_watchlist = "GOOG, TMUS, SKHY, NVDA, TSLA, AMD, SPY, QQQ"
user_tickers = st.sidebar.text_area("Watchlist (Comma Separated)", value=default_watchlist)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

if st.sidebar.button("🔄 Refresh Market Scanner") or 'scan_data' not in st.session_state:
    with st.spinner("Fetching live option chains, midpoints & technicals..."):
        st.session_state.scan_data = fetch_all_tickers(tickers, target_dte, target_delta)

results_list = st.session_state.scan_data

if results_list:
    df_results = pd.DataFrame([{k: v for k, v in item.items() if k != 'df_history'} for item in results_list])
    
    # Top Metrics Bar
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
    st.dataframe(styled_df, use_container_width=True, height=350)

    # Detailed Position Execution & Plotly Chart Section
    st.markdown("---")
    st.subheader("Position Execution & Interactive Chart")
    
    selected_ticker = st.selectbox("Select Ticker to Analyze", df_results["Ticker"].unique())
    selected_data = next(item for item in results_list if item["Ticker"] == selected_ticker)
    
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    pcol1.metric("Current Price", f"${selected_data['Price']}")
    pcol2.metric("Target Strike Price", f"${selected_data['Target Strike']}")
    pcol3.metric("Live Bid-Ask Midpoint", f"${selected_data['Midpoint Premium ($/sh)']:.2f} / sh")
    pcol4.metric("Total Upfront Credit", f"${selected_data['Contract Credit ($)']:,.2f} / contract")

    # Interactive Price & RSI Chart
    hist_df = selected_data['df_history']
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_df.index,
        open=hist_df['Open'], high=hist_df['High'],
        low=hist_df['Low'], close=hist_df['Close'],
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df['SMA50'],
        line=dict(color='orange', width=1.5),
        name="50-Day SMA"
    ))
    fig.add_hline(
        y=selected_data['Target Strike'],
        line_dash="dash", line_color="green" if "CSP" in selected_data['Signal'] else "red",
        annotation_text=f"Target Strike: ${selected_data['Target Strike']}"
    )
    
    fig.update_layout(
        title=f"{selected_ticker} Interactive Technical Chart",
        template="plotly_dark",
        height=450,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No valid ticker data returned. Please verify your watchlist symbols.")
