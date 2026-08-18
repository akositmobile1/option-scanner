import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
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
# BLACK-SCHOLES GREEK ENGINE
# ==========================================
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculates Delta and Vega for an option using Black-Scholes.
    S = Current Stock Price
    K = Strike Price
    T = Time to Expiration in years (e.g., 21/365)
    r = Risk-free interest rate (e.g., 0.045 for 4.5%)
    sigma = Implied Volatility (e.g., 0.25 for 25%)
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
        
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0  # Vega per 1% change in IV
    return round(float(delta), 3), round(float(vega), 3)

# ==========================================
# SINGLE TICKER PROCESSOR
# ==========================================
def process_single_ticker(ticker, target_dte=21, target_delta=0.15):
    try:
        data = yf.Ticker(ticker)
        df = data.history(period="1y")
        
        # Enforce minimum trading history (e.g., handles recent IPOs)
        if len(df) < 15:
            return None

        close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
        daily_change_pct = ((close - prev_close) / prev_close) * 100

        # Technical Indicators: 50-day SMA & 14-day RSI
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        delta_df = df['Close'].diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])
        sma50 = float(df['SMA50'].iloc[-1]) if not pd.isna(df['SMA50'].iloc[-1]) else close

        # Historical Volatility Estimation (21-day annualized std dev)
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        volatility_est = float(log_returns.tail(21).std() * np.sqrt(252))
        if np.isnan(volatility_est) or volatility_est == 0:
            volatility_est = 0.25

        # DTE Multiplier Scaling
        dte_multiplier = 2.1 if target_dte <= 21 else 3.2

        # ==========================================
        # SMART SIGNAL ENGINE (WITH DAILY DROP GUARD)
        # ==========================================
        # CSP SIGNAL: RSI < 52 OR stock dropped > 2.0% today while holding 50-SMA support
        if (rsi < 52 or daily_change_pct <= -2.0) and close >= (sma50 * 0.98):
            signal = "🟢 SELL CSP"
            opt_type = "put"
            est_strike = close * (1 - (target_delta * volatility_est * (target_dte / 30.0)))
            est_premium = close * target_delta * (volatility_est * dte_multiplier)

        # CC SIGNAL: RSI >= 60 AND stock is NOT down heavily today (> -1.5%)
        elif rsi >= 60 and daily_change_pct > -1.5:
            signal = "🔴 SELL CC"
            opt_type = "call"
            est_strike = close * (1 + (target_delta * volatility_est * (target_dte / 30.0)))
            est_premium = close * target_delta * (volatility_est * dte_multiplier)

        # WAIT SIGNAL: Neutral zone or stock currently in a sharp daily pull-back
        else:
            signal = "⚪ WAIT"
            opt_type = "put"
            est_strike = close * (1 - (target_delta * volatility_est * (target_dte / 30.0)))
            est_premium = close * target_delta * (volatility_est * dte_multiplier)

        # Black-Scholes Greek Calculations
        T_years = target_dte / 365.0
        delta_val, vega_val = calculate_greeks(
            S=close,
            K=est_strike,
            T=T_years,
            r=0.045,  # 4.5% Risk-free rate benchmark
            sigma=volatility_est,
            option_type=opt_type
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

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.title("🎯 Options Yield & Portfolio Scanner")
st.caption("14–21 DTE Engine with Daily Drop Guards & Black-Scholes Greeks")

# Sidebar Configuration
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
    # Summary Metrics
    csp_count = len(df_results[df_results["Signal"] == "🟢 SELL CSP"])
    cc_count = len(df_results[df_results["Signal"] == "🔴 SELL CC"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Watchlist Count", len(df_results))
    col2.metric("CSP Opportunities", f"{csp_count} Tickers")
    col3.metric("CC Opportunities", f"{cc_count} Tickers")
    col4.metric("Max Position Limit (20%)", f"${portfolio_size * 0.20:,.0f}")

    st.markdown("---")

    # Main Data Table
    st.subheader("Live Scanner Results")
    
    # Apply row highlighting for scannability
    def highlight_signals(val):
        if val == "🟢 SELL CSP":
            return "background-color: #113824; color: #4EFE96;"
        elif val == "🔴 SELL CC":
            return "background-color: #4A151B; color: #FF6B6B;"
        return "color: #888888;"

    styled_df = df_results.style.map(highlight_signals, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Detailed Position Planner
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

portfolio_size = st.sidebar.number_input("Total Portfolio ($)", value=600000, step=25000)
max_collateral_pct = st.sidebar.slider("Max Collateral per Ticker (%)", 5, 30, 20) / 100.0
target_dte = st.sidebar.slider("Target DTE", 14, 60, 35)
delta_offset = st.sidebar.slider("Delta Target (Lower = Safer)", 0.05, 0.30, 0.15)

watchlist_input = st.sidebar.text_area(
    "Watchlist Tickers (Comma Separated)",
    value="TSLA, NVDA, AMD, CRWV, NU, SNOW, WDAY, SKHY, NBIS, BABA, MSTR, IREN, TMUS, GOOG, AMZN, META, AAPL, COIN, PLTR",
    height=100
)
watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

if st.sidebar.button("🔄 Force Live Data Refresh"):
    st.cache_data.clear()

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
        
    # 2. Moving Average Safety (Max 30 pts)
    dist_20 = row.get("Dist_20SMA_%", -10)
    dist_50 = row.get("Dist_50SMA_%", -10)
    
    if 0 <= dist_20 <= 5 and dist_50 > 0:
        score += 30  # Pullback in structural uptrend
    elif dist_20 > 5 and dist_50 > 0:
        score += 20  # Strong uptrend momentum
    elif dist_50 < 0:
        score += 0   # Downtrend penalty

    # 3. RSI Entry Timing (Max 20 pts)
    rsi = row.get("RSI", 50)
    if 35 <= rsi <= 52:
        score += 20  # Ideal dip buy for CSP
    elif 52 < rsi <= 60:
        score += 10  # Neutral
    elif rsi > 65 or rsi < 30:
        score += 0   # Overbought peak or severe breakdown

    # 4. Active Signal Match Bonus (Max 10 pts)
    if row.get("Signal") in ["🟢 SELL CSP", "🔴 SELL CC"]:
        score += 10
        
    return score

def process_single_ticker(ticker, portfolio_size, max_collateral_pct, delta_offset, target_dte):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        
        # Requires 15 days minimum (accommodates recent IPOs/listings)
        if df.empty or len(df) < 15:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = float(df["Close"].iloc[-1])

        # Moving Averages & 14-day RSI
        df["20_SMA"] = df["Close"].rolling(20).mean()
        df["50_SMA"] = df["Close"].rolling(50).mean()
        df["RSI"] = calculate_rsi(df["Close"], period=14)

        # Fallbacks for younger assets
        sma20 = float(df["20_SMA"].iloc[-1]) if len(df) >= 20 else close
        sma50 = float(df["50_SMA"].iloc[-1]) if len(df) >= 50 else close
        rsi = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else 50.0

        dist_20 = (((close - sma20) / sma20) * 100) if len(df) >= 20 else 0.0
        dist_50 = (((close - sma50) / sma50) * 100) if len(df) >= 50 else 0.0

        # Volatility via 14-day ATR average
        daily_pct_change = df["Close"].pct_change().abs()
        volatility_est = float(daily_pct_change.tail(14).mean())

        # ==========================================
        # CALIBRATED SIGNAL ENGINE (ADJUSTED)
        # ==========================================
        # CSP Trigger: Shallow dip (RSI < 52) while holding structural support (within 2% of 50-SMA)
        if rsi < 52 and close >= (sma50 * 0.98):
            signal = "🟢 SELL CSP"
            est_strike = close * (1 - (delta_offset * volatility_est * 15))
            est_premium = close * delta_offset * (volatility_est * 3.2)
        # CC Trigger: RSI above 60 (Overbought / Momentum Peak)
        elif rsi >= 60:
            signal = "🔴 SELL CC"
            est_strike = close * (1 + (delta_offset * volatility_est * 15))
            est_premium = close * delta_offset * (volatility_est * 3.2)
        # WAIT Trigger: Neutral consolidation zone
        else:
            signal = "⚪ WAIT"
            est_strike = close * (1 - (delta_offset * volatility_est * 15))
            est_premium = close * delta_offset * (volatility_est * 3.2)

        # Sizing and Annualized Yield
        max_collateral_per_trade = portfolio_size * max_collateral_pct
        collateral_req = est_strike * 100
        contracts = max(1, int(max_collateral_per_trade // collateral_req)) if collateral_req > 0 else 1
        total_collateral = contracts * collateral_req
        total_est_credit = contracts * est_premium * 100

        roc_35_days = ((est_premium * 100) / collateral_req) if collateral_req > 0 else 0
        ann_roc = roc_35_days * (365 / target_dte) * 100

        return {
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
        }
    except Exception:
        return None

# 60-second Cache TTL ensures fresh updates on market shifts
@st.cache_data(ttl=60)
def fetch_and_analyze_data(tickers, portfolio_size, max_collateral_pct, delta_offset, target_dte):
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                process_single_ticker,
                ticker,
                portfolio_size,
                max_collateral_pct,
                delta_offset,
                target_dte
            ) for ticker in tickers
        ]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["Score"] = res_df.apply(calculate_option_score, axis=1)
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        
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
with st.spinner("Executing parallel scanning engine..."):
    scanner_df = fetch_and_analyze_data(
        watchlist,
        portfolio_size,
        max_collateral_pct,
        delta_offset,
        target_dte
    )

if scanner_df.empty:
    st.warning("No ticker data loaded. Please verify ticker list or connection.")
else:
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
    # 5. SINGLE-TICKER DEEP DIVE & OPTION CHAIN
    # ==========================================
    st.subheader("🔍 Single-Ticker Deep Dive & Live Option Chain")

    selected_ticker = st.selectbox(
        "Select Ticker to Inspect Live Option Chain",
        options=scanner_df["Ticker"].tolist()
    )

    ticker_row = scanner_df[scanner_df["Ticker"] == selected_ticker].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Action", ticker_row["Signal"])
    c2.metric("Target Strike", f"${ticker_row['Est_Strike']:.2f}")
    c3.metric("Max Contracts", f"{ticker_row['Contracts']} (${ticker_row['Total_Collateral']:,.0f} Collateral)")
    c4.metric("Est. Total Credit", f"${ticker_row['Total_Credit']:,.2f}")

    st.caption(f"Real-Time Technical Chart for {selected_ticker}")
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:400px;width:100%;">
      <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_1&symbol={selected_ticker}&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=RSI@tv-basicstudies%2CMAExp@tv-basicstudies&theme=dark&style=1&timezone=Etc%2FUTC" 
              style="width: 100%; height: 100%; border: none;"></iframe>
    </div>
    """
    st.components.v1.html(tradingview_html, height=410)

    st.markdown(f"### ⚡ Live Option Chain ({selected_ticker})")

    try:
        tk = yf.Ticker(selected_ticker)
        expirations = tk.options

        if expirations:
            exp_choice = st.selectbox("Select Expiration Date", options=expirations[:6])
            chain = tk.option_chain(exp_choice)

            tab1, tab2 = st.tabs(["Put Option Chain (CSPs)", "Call Option Chain (CCs)"])

            with tab1:
                puts_df = chain.puts[['strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest']].copy()
                puts_df['midPrice'] = (puts_df['bid'] + puts_df['ask']) / 2.0
                puts_df['impliedVolatility'] = puts_df['impliedVolatility'] * 100
                st.dataframe(
                    puts_df[['strike', 'bid', 'midPrice', 'ask', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest']],
                    column_config={
                        "strike": "Strike ($)",
                        "bid": "Bid ($)",
                        "midPrice": st.column_config.NumberColumn("Mid ($)", format="$%.2f"),
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
                calls_df['midPrice'] = (calls_df['bid'] + calls_df['ask']) / 2.0
                calls_df['impliedVolatility'] = calls_df['impliedVolatility'] * 100
                st.dataframe(
                    calls_df[['strike', 'bid', 'midPrice', 'ask', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest']],
                    column_config={
                        "strike": "Strike ($)",
                        "bid": "Bid ($)",
                        "midPrice": st.column_config.NumberColumn("Mid ($)", format="$%.2f"),
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
            st.info(f"No option chain data available for {selected_ticker}.")
    except Exception as e:
        st.error(f"Error reading option chain: {e}")
