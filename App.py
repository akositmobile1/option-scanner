import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(
    page_title="Options Income Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    "NBIS",
    "AMZN",
    "CRWV",
    "NVDA",
    "CBRS",
    "PLTR",
    "SKHY",
    "ENPH",
    "AGNC",
    "GOOG",
    "MSTR",
    "COIN",
    "SMCI",
]

st.sidebar.header("💰 Portfolio Settings")
total_capital = st.sidebar.number_input(
    "Total Capital ($)", value=600000, step=25000, key="capital_input"
)
weekly_target = st.sidebar.number_input(
    "Weekly Target ($)", value=3000, step=250, key="target_input"
)
target_delta = st.sidebar.slider(
    "Delta Offset", 0.05, 0.20, 0.08, 0.01, key="delta_slider"
)
max_alloc_pct = (
    st.sidebar.slider(
        "Max Collateral per Stock (%)", 10, 35, 20, key="alloc_slider"
    )
    / 100.0
)

selected_tickers = st.sidebar.multiselect(
    "Watchlist",
    DEFAULT_WATCHLIST,
    default=DEFAULT_WATCHLIST,
    key="watchlist_v9",
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

        daily_range = (df["High"] - df["Low"]).tail(14).mean()
        volatility_pct = daily_range / current_price

        target_put_strike = current_price * (
            1 - (delta_offset * volatility_pct * 15)
        )
        target_call_strike = current_price * (
            1 + (delta_offset * volatility_pct * 15)
        )

        est_contract_premium = (
            current_price * delta_offset * (volatility_pct * 12)
        )
        est_weekly_premium_per_contract = est_contract_premium / 4.0

        max_collateral = capital * max_alloc
        max_contracts = max(1, int(max_collateral / (current_price * 100)))
        total_weekly_est = (
            est_weekly_premium_per_contract * 100 * max_contracts
        )

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


if selected_tickers:
    df_results = run_pro_scanner(
        selected_tickers,
        total_capital,
        weekly_target,
        target_delta,
        max_alloc_pct,
    )
    st.dataframe(df_results, use_container_width=True, hide_index=True)

# Technical Chart Section via TradingView Embed
st.markdown("---")
st.markdown(
    "<h3 style='font-size: 1.1rem; margin-bottom: 0px;'>📈 Technical Chart Inspector</h3>",
    unsafe_allow_html=True,
)
chart_symbol = st.selectbox(
    "Select Ticker to Inspect", selected_tickers, key="chart_select"
)

if chart_symbol:
    tv_code = f"""
    <div class="tradingview-widget-container" style="height:450px;width:100%;">
      <div id="tradingview_chart" style="height:450px;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{chart_symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tv_code, height=460)
