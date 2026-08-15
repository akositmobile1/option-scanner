import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Pro Options Scanner ($3k/Wk Target)", layout="wide"
)
st.title("🎯 Pro Options Income Engine & Position Calculator")

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
    "Total Portfolio Capital ($)", value=600000, step=25000
)
weekly_target = st.sidebar.number_input(
    "Weekly Income Target ($)", value=3000, step=250
)
target_delta = st.sidebar.slider(
    "Target Option Delta Offset", 0.05, 0.20, 0.08, 0.01
)
max_alloc_pct = (
    st.sidebar.slider("Max Collateral per Stock (%)", 10, 35, 20) / 100.0
)

st.sidebar.header("📊 Technical Parameters")
selected_tickers = st.sidebar.multiselect(
    "Active Watchlist",
    DEFAULT_WATCHLIST,
    default=DEFAULT_WATCHLIST,
    key="watchlist_v2",
)


@st.cache_data(ttl=300)
def run_pro_scanner(tickers, capital, target, delta_offset, max_alloc):
    results = []

    for symbol in tickers:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
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

        # 4. Position Sizing & Collateral Caps (20% Max per Ticker)
        max_collateral = capital * max_alloc
        max_contracts = max(1, int(max_collateral / (current_price * 100)))
        total_weekly_est = (
            est_weekly_premium_per_contract * 100 * max_contracts
        )

        # Technical Indicators: RSI (14) & SMAs (20/50)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        sma_fast = float(df["Close"].rolling(20).mean().iloc[-1])
        sma_slow = float(df["Close"].rolling(50).mean().iloc[-1])

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
                "RSI (14)": round(rsi, 1),
                "Signal": signal,
                "Target Put Strike": round(target_put_strike, 2),
                "Target Call Strike": round(target_call_strike, 2),
                "Max Contracts": max_contracts,
                "Max Collateral": f"${round(max_contracts * current_price * 100, 2):,}",
                "Est. Weekly Income": f"${round(total_weekly_est, 2)}",
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
    st.dataframe(df_results, use_container_width=True)
