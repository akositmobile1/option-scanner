import pandas as pd
import streamlit as st
import yfinance as yf

# Page setup optimized for mobile responsiveness
st.set_page_config(
    page_title="Options Income Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Mobile Styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
    }
    .mobile-title {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stDataFrame"] {
        font-size: 0.82rem !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 class='mobile-title'>🎯 Options Income Engine</h2>",
    unsafe_allow_html=True,
)

DEFAULT_WATCHLIST = [
    "TMUS",
    "IREN",
    "TSLA",
    "SNOW",
    "RDDT",
    "AMD",
    "NVDA",
    "PLTR",
    "SKHY",
    "ENPH",
    "AGNC",
    "GOOG",
    "MSTR",
    "COIN",
    "SMCI",
]

# Sidebar Controls
st.sidebar.header("💰 Portfolio Settings")
total_capital = st.sidebar.number_input(
    "Total Capital ($)", value=600000, step=25000
)
weekly_target = st.sidebar.number_input(
    "Weekly Target ($)", value=3000, step=250
)
target_delta = st.sidebar.slider(
    "Delta Offset", 0.05, 0.20, 0.08, 0.01
)
max_alloc_pct = (
    st.sidebar.slider("Max Collateral per Stock (%)", 10, 35, 20) / 100.0
)

selected_tickers = st.sidebar.multiselect(
    "Watchlist",
    DEFAULT_WATCHLIST,
    default=DEFAULT_WATCHLIST,
    key="watchlist_v7",
)


@st.cache_data(ttl=300)
def run_pro_scanner(tickers, capital, target, delta_offset, max_alloc):
    results = []

    for symbol in tickers:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or "Close" not in df.columns:
            continue

        current_price = float(df["Close"].iloc[-1])

        # Daily Volatility Calculation
        daily_range = (df["High"] - df["Low"]).tail(14).mean()
        volatility_pct = daily_range / current_price

        # Volatility-Adjusted Target Strikes (~0.18 Delta Target)
        target_put_strike = current_price * (
            1 - (delta_offset * volatility_pct * 15)
        )
        target_call_strike = current_price * (
            1 + (delta_offset * volatility_pct * 15)
        )

        # Premium & Position Sizing
        est_contract_premium = (
            current_price * delta_offset * (volatility_pct * 12)
        )
        est_weekly_premium_per_contract = est_contract_premium / 4.0

        max_collateral = capital * max_alloc
        max_contracts = max(1, int(max_collateral / (current_price * 100)))
        total_weekly_est = (
            est_weekly_premium_per_contract * 100 * max_contracts
        )

        # RSI & Signal Logic
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        sma_fast = float(df["Close"].rolling(20).mean().iloc[-1])

        signal = "NEUTRAL"
        if rsi < 40 and current_price <= sma_fast:
            signal = "🟢 SELL CSP"
        elif rsi > 60 and current_price >= sma_fast:
            signal = "🔴 SELL CC"

        results.append(
            {
                "Ticker": symbol,
                "Price": round(current_price, 2),
                "RSI": round(rsi, 1),
                "Signal": signal,
                "Put Strike": round(target_put_strike, 2),
                "Call Strike": round(target_call_strike, 2),
                "Contracts": max_contracts,
                "Collateral": f"${round(max_contracts * current_price * 100, 0):,.0f}",
                "Est. Wk Income": f"${round(total_weekly_est, 2)}",
            }
        )

    return pd.DataFrame(results)


# Render Table
if selected_tickers:
    df_results = run_pro_scanner(
        selected_tickers,
        total_capital,
        weekly_target,
        target_delta,
        max_alloc_pct,
    )
    st.dataframe(df_results, use_container_width=True, hide_index=True)

# Technical Chart Section
st.markdown("---")
st.markdown(
    "<h3 style='font-size: 1.1rem; margin-bottom: 0px;'>📈 Technical Chart Inspector</h3>",
    unsafe_allow_html=True,
)
chart_symbol = st.selectbox("Select Ticker to Inspect", selected_tickers)

if chart_symbol:
    try:
        df_chart = yf.download(
            chart_symbol, period="6m", interval="1d", progress=False
        )
        if isinstance(df_chart.columns, pd.MultiIndex):
            df_chart.columns = df_chart.columns.get_level_values(0)

        if not df_chart.empty and "Close" in df_chart.columns:
            chart_data = pd.DataFrame(
                {
                    "Price": df_chart["Close"],
                    "20 SMA": df_chart["Close"].rolling(20).mean(),
                },
                index=df_chart.index,
            )
            st.line_chart(chart_data)
        else:
            st.warning(f"No price data available for {chart_symbol}.")
    except Exception as e:
        st.error(f"Error loading chart for {chart_symbol}: {e}")

st.markdown(
    "<h2 class='mobile-title'>🎯 Options Income Engine</h2>",
    unsafe_allow_html=True,
)

DEFAULT_WATCHLIST = [
    "TMUS",
    "IREN",
    "TSLA",
    "SNOW",
    "RDDT",
    "AMD",
    "NVDA",
    "PLTR",
    "SKHY",
    "ENPH",
    "AGNC",
    "GOOG",
    "MSTR",
    "COIN",
    "SMCI",
]

# Sidebar Controls
st.sidebar.header("💰 Portfolio Settings")
total_capital = st.sidebar.number_input(
    "Total Capital ($)", value=600000, step=25000
)
weekly_target = st.sidebar.number_input(
    "Weekly Target ($)", value=3000, step=250
)
target_delta = st.sidebar.slider(
    "Delta Offset", 0.05, 0.20, 0.08, 0.01
)
max_alloc_pct = (
    st.sidebar.slider("Max Collateral per Stock (%)", 10, 35, 20) / 100.0
)

selected_tickers = st.sidebar.multiselect(
    "Watchlist",
    DEFAULT_WATCHLIST,
    default=DEFAULT_WATCHLIST,
    key="watchlist_v6",
)


@st.cache_data(ttl=300)
def run_pro_scanner(tickers, capital, target, delta_offset, max_alloc):
    results = []

    for symbol in tickers:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or "Close" not in df.columns:
            continue

        current_price = float(df["Close"].iloc[-1])

        # Daily Volatility Calculation
        daily_range = (df["High"] - df["Low"]).tail(14).mean()
        volatility_pct = daily_range / current_price

        # Volatility-Adjusted Target Strikes
        target_put_strike = current_price * (
            1 - (delta_offset * volatility_pct * 15)
        )
        target_call_strike = current_price * (
            1 + (delta_offset * volatility_pct * 15)
        )

        # Premium & Sizing
        est_contract_premium = (
            current_price * delta_offset * (volatility_pct * 12)
        )
        est_weekly_premium_per_contract = est_contract_premium / 4.0

        max_collateral = capital * max_alloc
        max_contracts = max(1, int(max_collateral / (current_price * 100)))
        total_weekly_est = (
            est_weekly_premium_per_contract * 100 * max_contracts
        )

        # RSI & Signal
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        sma_fast = float(df["Close"].rolling(20).mean().iloc[-1])

        signal = "NEUTRAL"
        if rsi < 40 and current_price <= sma_fast:
            signal = "🟢 SELL CSP"
        elif rsi > 60 and current_price >= sma_fast:
            signal = "🔴 SELL CC"

        results.append(
            {
                "Ticker": symbol,
                "Price": round(current_price, 2),
                "RSI": round(rsi, 1),
                "Signal": signal,
                "Put Strike": round(target_put_strike, 2),
                "Call Strike": round(target_call_strike, 2),
                "Contracts": max_contracts,
                "Collateral": f"${round(max_contracts * current_price * 100, 0):,.0f}",
                "Est. Wk Income": f"${round(total_weekly_est, 2)}",
            }
        )

    return pd.DataFrame(results)


# Render Table
if selected_tickers:
    df_results = run_pro_scanner(
        selected_tickers,
        total_capital,
        weekly_target,
        target_delta,
        max_alloc_pct,
    )
    st.dataframe(df_results, use_container_width=True, hide_index=True)

# Technical Chart Section
st.markdown("---")
st.markdown(
    "<h3 style='font-size: 1.1rem; margin-bottom: 0px;'>📈 Technical Chart Inspector</h3>",
    unsafe_allow_html=True,
)
chart_symbol = st.selectbox("Select Ticker to Inspect", selected_tickers)

if chart_symbol:
    df_chart = yf.download(
        chart_symbol, period="6m", interval="1d", progress=False
    )
    if isinstance(df_chart.columns, pd.MultiIndex):
        df_chart.columns = df_chart.columns.get_level_values(0)

    if not df_chart.empty and "Close" in df_chart.columns:
        # Prepare clean Dataframe for native Streamlit line chart
        chart_data = pd.DataFrame()
        chart_data["Price"] = df_chart["Close"]
        chart_data["20 SMA"] = df_chart["Close"].rolling(20).mean()

        # Native lightweight mobile chart rendering
        st.line_chart(chart_data, color=["#00FFAA", "#FF9900"])
