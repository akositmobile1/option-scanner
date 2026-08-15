import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Options & Swing Strategy Scanner", layout="wide"
)
st.title("🎯 Options Entry/Exit Scanner & Technical Dashboard")

# Core Watchlist
WATCHLIST = [
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
st.sidebar.header("Scan Parameters")
selected_tickers = st.sidebar.multiselect(
    "Active Watchlist", WATCHLIST, default=WATCHLIST
)
timeframe = st.sidebar.selectbox("Chart Timeframe", ["1d", "1h"], index=0)
min_market_cap_b = st.sidebar.number_input(
    "Min Market Cap ($B)", value=2.0, step=1.0
)
sma_fast_len = st.sidebar.slider("Fast Moving Average (SMA)", 5, 20, 10)
sma_slow_len = st.sidebar.slider("Slow Moving Average (SMA)", 20, 50, 20)


def calculate_rsi(series, period=14):
  """Calculates standard Relative Strength Index (RSI)."""
  delta = series.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))


def get_atm_iv(ticker_obj, spot_price):
  """Extracts At-The-Money (ATM) Implied Volatility from the nearest option chain."""
  try:
    expirations = ticker_obj.options
    if not expirations:
      return 0.0
    chain = ticker_obj.option_chain(expirations[0])
    calls = chain.calls
    if calls.empty:
      return 0.0
    calls["diff"] = (calls["strike"] - spot_price).abs()
    atm_contract = calls.sort_values("diff").iloc[0]
    return float(atm_contract["impliedVolatility"]) * 100
  except Exception:
    return 0.0


@st.cache_data(ttl=300)
def run_scanner(tickers, tf, min_cap):
  results = []
  period = "1y" if tf == "1d" else "1mo"

  for symbol in tickers:
    tk = yf.Ticker(symbol)
    df = tk.history(period=period, interval=tf)

    if df.empty or len(df) < sma_slow_len:
      continue

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    info = tk.info or {}
    mkt_cap = info.get("marketCap", 0) / 1e9

    if mkt_cap < min_cap and mkt_cap > 0:
      continue

    # Pure Pandas Technicals
    df["RSI"] = calculate_rsi(df["Close"], 14)
    df["SMA_Fast"] = df["Close"].rolling(window=sma_fast_len).mean()
    df["SMA_Slow"] = df["Close"].rolling(window=sma_slow_len).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close_price = latest["Close"]
    rsi_val = latest["RSI"]
    fast_val = latest["SMA_Fast"]
    slow_val = latest["SMA_Slow"]

    atm_iv = get_atm_iv(tk, close_price)

    put_strike_est = close_price * 0.92
    call_strike_est = close_price * 1.08

    bullish_cross = (prev["SMA_Fast"] <= prev["SMA_Slow"]) and (
        fast_val > slow_val
    )
    bearish_cross = (prev["SMA_Fast"] >= prev["SMA_Slow"]) and (
        fast_val < slow_val
    )

    if rsi_val < 38 or bullish_cross:
      signal = "🟢 SELL CSP / BUY SHARES"
    elif rsi_val > 68 or bearish_cross:
      signal = "🔴 SELL COVERED CALL / TAKE PROFIT"
    else:
      signal = "⚪ NEUTRAL / HOLD"

    results.append({
        "Ticker": symbol,
        "Price": f"${close_price:.2f}",
        "Market Cap ($B)": (
            f"${mkt_cap:.1f}B" if mkt_cap > 0 else "N/A"
        ),
        "ATM IV (%)": f"{atm_iv:.1f}%" if atm_iv > 0 else "N/A",
        "RSI (14)": (
            f"{rsi_val:.1f}" if pd.notnull(rsi_val) else "N/A"
        ),
        "Est. Put Strike (~0.18 Delta)": f"${put_strike_est:.2f}",
        "Est. Call Strike (~0.18 Delta)": f"${call_strike_est:.2f}",
        "Signal": signal,
    })

  return pd.DataFrame(results)


# Render Table
if selected_tickers:
  with st.spinner("Scanning market data and option chains..."):
    scan_results = run_scanner(selected_tickers, timeframe, min_market_cap_b)

  st.subheader("Market Scan & Options Strike Targets")
  st.dataframe(scan_results, use_container_width=True)

  # Charting View
  selected_stock = st.selectbox("Inspect Ticker Chart", selected_tickers)
  if selected_stock:
    tk = yf.Ticker(selected_stock)
    history_period = "1y" if timeframe == "1d" else "1mo"
    chart_df = tk.history(period=history_period, interval=timeframe)

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
          title=f"{selected_stock} Technical Chart",
          xaxis_rangeslider_visible=False,
          height=500,
      )
      st.plotly_chart(fig, use_container_width=True)
