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
    "Active Watchlist", DEFAULT_WATCHLIST, default=DEFAULT_WATCHLIST
)
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h"], index=0)
sma_fast_len = st.sidebar.slider("Fast SMA", 5, 20, 10)
sma_slow_len = st.sidebar.slider("Slow SMA", 20, 50, 20)


def calculate_rsi(series, period=14):
  delta = series.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))


@st.cache_data(ttl=300)
def run_pro_scanner(tickers, capital, target, delta_offset, max_alloc):
  results = []

  for symbol in tickers:
    tk = yf.Ticker(symbol)
    df = tk.history(period="1y", interval="1d")

    if df.empty or len(df) < sma_slow_len:
      continue

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    close_price = df["Close"].iloc[-1]
    df["RSI"] = calculate_rsi(df["Close"], 14)
    rsi_val = df["RSI"].iloc[-1]

    # Strike Estimates
    put_strike = round(close_price * (1 - delta_offset), 2)
    call_strike = round(close_price * (1 + delta_offset), 2)

    # Position Sizing Math
    max_capital_ticker = capital * max_alloc
    collateral_per_put = put_strike * 100
    rec_contracts = (
        int(max_capital_ticker // collateral_per_put)
        if collateral_per_put > 0
        else 0
    )
    collateral_used = rec_contracts * collateral_per_put

    # Premium & Return Estimates (~0.5% - 1.0% Weekly Yield Proxy)
    est_weekly_premium_per_contract = collateral_per_put * 0.006
    est_weekly_income = rec_contracts * est_weekly_premium_per_contract

    # Signal Rules
    if symbol == "TMUS":
      if rsi_val < 45:
        signal = "🟢 SELL CSP (TMUS Dip)"
      elif rsi_val > 62:
        signal = "🔴 SELL COVERED CALL"
      else:
        signal = "⚪ NEUTRAL (Yield Focus)"
    else:
      if rsi_val < 38:
        signal = "🟢 SELL CSP / BUY SHARES"
      elif rsi_val > 68:
        signal = "🔴 SELL COVERED CALL"
      else:
        signal = "⚪ NEUTRAL / HOLD"

    results.append({
        "Ticker": symbol,
        "Price": f"${close_price:.2f}",
        "RSI (14)": f"{rsi_val:.1f}" if pd.notnull(rsi_val) else "N/A",
        "Target Put (~0.18 Delta)": f"${put_strike:.2f}",
        "Target Call (~0.18 Delta)": f"${call_strike:.2f}",
        "Max Contracts": rec_contracts,
        "Required Collateral": f"${collateral_used:,.0f}",
        "Est. Weekly Income": f"${est_weekly_income:,.0f}",
        "Signal": signal,
    })

  return pd.DataFrame(results)


# Render Portfolio Feasibility Metrics
required_annual_yield = (
    ((weekly_target * 52) / total_capital) * 100 if total_capital > 0 else 0
)

st.subheader("💡 Target Feasibility Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Weekly Goal", f"${weekly_target:,.0f}/wk")
c2.metric("Required Capital Base", f"${total_capital:,.0f}")
c3.metric("Target Annual Yield", f"{required_annual_yield:.1f}%")

st.markdown("---")

# Render Main Scanner Table
if selected_tickers:
  with st.spinner("Scanning market data and computing position sizes..."):
    scan_results = run_pro_scanner(
        selected_tickers,
        total_capital,
        weekly_target,
        target_delta,
        max_alloc_pct,
    )

  st.subheader("📊 Trade Signals & Position Sizing Engine")
  st.dataframe(scan_results, use_container_width=True)

  # Charting Component
  selected_stock = st.selectbox("Inspect Technical Chart", selected_tickers)
  if selected_stock:
    tk = yf.Ticker(selected_stock)
    chart_df = tk.history(period="1y", interval="1d")

    if not chart_df.empty:
      if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.get_level_values(0)

      chart_df["SMA_Fast"] = (
          chart_df["Close"].rolling(window=sma_fast_len).mean()
      )
      chart_df["SMA_Slow"] = (
          chart_df["Close"].rolling(window=sma_slow_len).mean()
      )

      fig = go.Figure()
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
      fig.add_trace(
          go.Scatter(
              x=chart_df.index,
              y=chart_df["SMA_Fast"],
              line=dict(color="orange", width=1.5),
              name=f"SMA {sma_fast_len}",
          )
      )
      fig.add_trace(
          go.Scatter(
              x=chart_df.index,
              y=chart_df["SMA_Slow"],
              line=dict(color="blue", width=1.5),
              name=f"SMA {sma_slow_len}",
          )
      )

      fig.update_layout(
          title=f"{selected_stock} Interactive Chart",
          xaxis_rangeslider_visible=False,
          height=500,
      )
      st.plotly_chart(fig, use_container_width=True)
