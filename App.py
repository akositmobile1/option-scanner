import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page setup with mobile responsiveness optimizations
st.set_page_config(
    page_title="Options Income Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Mobile Responsive Styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    .mobile-title {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    [data-testid="stDataFrame"] {
        font-size: 0.85rem !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Compact Header
st.markdown(
    "<h2 class='mobile-title'>🎯 Options Income Engine</h2>",
    unsafe_allow_html=True,
)

# Core Watchlist
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
st.sidebar.header("💰 Portfolio & Target Settings")
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

st.sidebar.header("📊 Active Tickers")
selected_tickers = st.sidebar.multiselect(
    "Watchlist",
    DEFAULT_WATCHLIST,
    default=DEFAULT_WATCHLIST,
    key="watchlist_v4",
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

        # 1. Calculate Daily Volatility Percentage
        daily_range = (df["High"] - df["Low"]).tail(14).mean()
        volatility_pct = daily_range / current_price

        # 2. Volatility-Adjusted Target Strikes (~0.18 Delta Target)
        target_put_strike = current_price * (
            1 - (delta_offset * volatility_pct * 15)
        )
        target_call_strike = current_price * (
            1 + (delta_offset * volatility_pct * 15)
        )

        # 3. Volatility-Scaled Premium Calculation
        est_contract_premium = (
            current_price * delta_offset * (volatility_pct * 12)
        )
        est_weekly_premium_per_contract = est_contract_premium / 4.0

        # 4. Position Sizing & Collateral Caps
        max_collateral = capital * max_alloc
        max_contracts = max(1, int(max_collateral / (current_price * 100)))
        total_weekly_est = (
            est_weekly_premium_per_contract * 100 * max_contracts
        )

        # Technical Indicators: RSI (14) & SMA (20)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        sma_fast = float(df["Close"].rolling(20).mean().iloc[-1])

        # Trade Signal Logic
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


# Run Scanner & Display Table
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
    chart_df = yf.download(
        chart_symbol, period="6m", interval="1d", progress=False
    )

    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.get_level_values(0)

    if not chart_df.empty and "Close" in chart_df.columns:
        chart_df["SMA20"] = chart_df["Close"].rolling(20).mean()

        fig = go.Figure()

        # Candlestick Trace
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="Price",
            )
        )

        # 20 SMA Overlay
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["SMA20"],
                mode="lines",
                name="20 SMA",
                line=dict(color="#FF9900", width=1.5),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=5, r=5, t=10, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
