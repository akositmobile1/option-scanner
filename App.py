import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==========================================
# 1. SAFE APP CONFIGURATION
# ==========================================
try:
    st.set_page_config(
        page_title="Options Income Radar",
        page_layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

# Custom Mobile-Friendly CSS
st.markdown(
    """
    <style>
    .main { padding: 0rem 0.5rem; }
    .stDataFrame { width: 100%; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚡ Options Income Radar")
st.caption("Targeting $2k-$6k/week via 30-45 DTE Cash-Secured Puts & Covered Calls")

# ==========================================
# 2. SIDEBAR PARAMETERS
# ==========================================
st.sidebar.header("🎯 Strategy Settings")

portfolio_size = st.sidebar.number_input("Total Portfolio ($)", value=600000, step=25000)
max_collateral_pct = st.sidebar.slider("Max Collateral per Ticker (%)", 5, 30, 20) / 100.0
target_dte = st.sidebar.slider("Target DTE", 14, 60, 35)
delta_offset = st.sidebar.slider("Delta Target (Lower = Safer)", 0.05, 0.30, 0.15)

watchlist_input = st.sidebar.text_area(
    "Watchlist Tickers (Comma Separated)",
    value="TSLA, NVDA, AMD, MSTR, IREN, TMUS, GOOG, AMZN, META, AAPL, COIN, PLTR",
    height=100
)
watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

# ==========================================
# 3. TECHNICAL & SCORING CALCULATIONS
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_option_score(row):
    score = 0
    
    # 1. Annualized Yield / Return on Capital (Max 40 pts)
    ann_roc = row.get("Annualized_ROC_%", 0)
    if ann_roc >= 35:
        score += 40
    elif ann_roc >= 25:
        score += 30
    elif ann_roc >= 15:
        score += 15
        
    # 2. Trend & Moving Average Safety (Max 30 pts)
    dist_20 = row.get("Dist_20SMA_%", -10)
    dist_50 = row.get("Dist_50SMA_%", -10)
    
    if 0 <= dist_20 <= 5 and dist_50 > 0:
        score += 30  # Perfect pullback in structural uptrend
    elif dist_20 > 5 and dist_50 > 0:
        score += 20  # Strong uptrend
    elif dist_50 < 0:
        score += 0   # Downtrend penalty

    # 3. RSI Entry Timing (Max 20 pts)
    rsi = row.get("RSI", 50)
    if 35 <= rsi <= 48:
        score += 20  # Ideal oversold dip for CSP
    elif 48 < rsi <= 60:
        score += 10  # Neutral
    elif rsi > 65 or rsi < 30:
        score += 0   # Overbought or severe breakdown

    # 4. Active Signal Match Bonus (Max 10 pts)
    if row.get("Signal") in ["🟢 SELL CSP", "🔴 SELL CC"]:
        score += 10
        
    return score

@st.cache_data(ttl=300)
def fetch_and_analyze_data(tickers):
    results = []
    max_collateral_per_trade = portfolio_size * max_collateral_pct

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if df.empty or len(df) < 50:
                continue

            # Handle multi-index columns if returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = float(df["Close"].iloc[-1])

            # Moving Averages & RSI
            df["20_SMA"] = df["Close"].rolling(20).mean()
            df["50_SMA"] = df["Close"].rolling(50).mean()
            df["RSI"] = calculate_rsi(df["Close"])

            sma20 = float(df["20_SMA"].iloc[-1])
            sma50 = float(df["50_SMA"].iloc[-1])
            rsi = float(df["RSI"].iloc[-1])

            dist_20 = ((close - sma20) / sma20) * 100
            dist_50 = ((close - sma50) / sma50) * 100

            # Volatility estimate based on recent daily ATR
            daily_pct_change = df["Close"].pct_change().abs()
            volatility_est = float(daily_pct_change.tail(14).mean())

            # Signal Generation & Calibrated Premium Multiplier (3.2x)
            if rsi < 48 and close >= sma50:
                signal = "🟢 SELL CSP"
                est_strike = close * (1 - (delta_offset * volatility_est * 15))
                est_premium = close * delta_offset * (volatility_est * 3.2)
            elif rsi > 62:
                signal = "🔴 SELL CC"
                est_strike = close * (1 + (delta_offset * volatility_est * 15))
                est_premium = close * delta_offset * (volatility_est * 3.2)
            else:
                signal = "⚪ WAIT"
                est_strike = close * (1 - (delta_offset * volatility_est * 15))
                est_premium = close * delta_offset * (volatility_est * 3.2)

            # Contract Sizing & Capital Efficiency
            collateral_req = est_strike * 100
            contracts = max(1, int(max_collateral_per_trade // collateral_req)) if collateral_req > 0 else 1
            total_collateral = contracts * collateral_req
            total_est_credit = contracts * est_premium * 100

            # Annualized ROC Calculation
            roc_35_days = ((est_premium * 100) / collateral_req) if collateral_req > 0 else 0
            ann_roc = roc_35_days * (365 / target_dte) * 100

            results.append({
                "Ticker": ticker,
                "Price": close,
                "Signal": signal,
                "RSI": rsi,
                "Dist_20SMA_%": dist_20,
                "Dist_50SMA_%": dist_50,
                "Est_Strike": est_strike,
                "Est_Premium": est_premium,
                "Contracts": contracts,
                "Total_Collateral": total_collateral,
                "Total_Credit": total_est_credit,
                "Annualized_ROC_%": ann_roc
            })
        except Exception:
            continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["Score"] = res_df.apply(calculate_option_score, axis=1)
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        
        # Assign Rank Badges
        badges = []
        for i in range(len(res_df)):
            if i == 0:
                badges.append("🥇 1st")
            elif i == 1:
                badges.append("🥈 2nd")
            elif i == 2:
                badges.append("🥉 3rd")
            else:
                badges.append(f"#{i+1}")
        res_df["Rank"] = badges

    return res_df

# ==========================================
# 4. MAIN DASHBOARD RENDER
# ==========================================
with st.spinner("Scanning market volatility and ranking setups..."):
    scanner_df = fetch_and_analyze_data(watchlist)

if scanner_df.empty:
    st.warning("No ticker data found. Check your watchlist symbols or network.")
else:
    # Metrics Summary Row
    active_signals = scanner_df[scanner_df["Signal"] != "⚪ WAIT"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Watchlist Scanned", f"{len(scanner_df)} Tickers")
    col2.metric("Active Signals", f"{len(active_signals)} Setups")
    top_pick = scanner_df.iloc[0]["Ticker"] if not scanner_df.empty else "None"
    col3.metric("Top Ranked Opportunity", top_pick)

    st.subheader("📊 Ranked Options Screener")

    display_cols = [
        "Rank", "Ticker", "Score", "Signal", "Price",
        "Est_Strike", "Est_Premium", "Annualized_ROC_%", "RSI", "Dist_20SMA_%"
    ]

    st.dataframe(
        scanner_df[display_cols],
        column_config={
            "Score": st.column_config.ProgressColumn("Match Score", min_value=0, max_value=100, format="%d pts"),
            "Price": st.column_config.NumberColumn("Stock Price", format="$%.2f"),
            "Est_Strike": st.column_config.NumberColumn("Est. Strike", format="$%.2f"),
            "Est_Premium": st.column_config.NumberColumn("Est. Premium", format="$%.2f"),
            "Annualized_ROC_%": st.column_config.NumberColumn("Ann. Yield", format="%.1f%%"),
            "Dist_20SMA_%": st.column_config.NumberColumn("20-SMA Dist", format="%.1f%%"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.1f")
        },
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # 5. SINGLE-TICKER LIVE INSPECTOR
    # ==========================================
    st.subheader("🔍 Single-Ticker Deep Dive & Live Option Chain")

    selected_ticker = st.selectbox(
        "Select Ticker to Inspect Live Option Chain",
        options=scanner_df["Ticker"].tolist()
    )

    ticker_row = scanner_df[scanner_df["Ticker"] == selected_ticker].iloc[0]

    # Trade Blueprint Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Action", ticker_row["Signal"])
    c2.metric("Target Strike", f"${ticker_row['Est_Strike']:.2f}")
    c3.metric("Max Contracts", f"{ticker_row['Contracts']} (${ticker_row['Total_Collateral']:,.0f} Collateral)")
    c4.metric("Est. Total Credit", f"${ticker_row['Total_Credit']:,.2f}")

    # TradingView Chart Widget
    st.caption(f"Real-Time Technical Chart for {selected_ticker}")
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:400px;width:100%;">
      <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_1&symbol={selected_ticker}&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=RSI@tv-basicstudies%2CMAExp@tv-basicstudies&theme=dark&style=1&timezone=Etc%2FUTC" 
              style="width: 100%; height: 100%; border: none;"></iframe>
    </div>
    """
    st.components.v1.html(tradingview_html, height=410)

    # Live Option Chain Fetcher (Single Ticker)
    st.markdown(f"### ⚡ Live Option Chain ({selected_ticker})")

    try:
        tk = yf.Ticker(selected_ticker)
        expirations = tk.options

        if expirations:
            exp_choice = st.selectbox("Select Live Expiration Date", options=expirations[:6])
            chain = tk.option_chain(exp_choice)

            tab1, tab2 = st.tabs(["Put Option Chain (CSPs)", "Call Option Chain (CCs)"])

            with tab1:
                puts_df = chain.puts[['strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest']].copy()
                puts_df['impliedVolatility'] = puts_df['impliedVolatility'] * 100
                st.dataframe(
                    puts_df,
                    column_config={
                        "strike": "Strike ($)",
                        "bid": "Bid ($)",
                        "ask": "Ask ($)",
                        "lastPrice": "Last ($)",
                        "impliedVolatility": st.column_config.NumberColumn("IV", format="%.1f%%"),
                        "volume": "Volume",
                        "openInterest": "Open Int"
                    },
                    hide_index=True,
                    use_container_width=True
                )

            with tab2:
                calls_df = chain.calls[['strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest']].copy()
                calls_df['impliedVolatility'] = calls_df['impliedVolatility'] * 100
                st.dataframe(
                    calls_df,
                    column_config={
                        "strike": "Strike ($)",
                        "bid": "Bid ($)",
                        "ask": "Ask ($)",
                        "lastPrice": "Last ($)",
                        "impliedVolatility": st.column_config.NumberColumn("IV", format="%.1f%%"),
                        "volume": "Volume",
                        "openInterest": "Open Int"
                    },
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info(f"No option chain data found for {selected_ticker}.")
    except Exception as e:
        st.error(f"Could not load live option chain: {e}")
