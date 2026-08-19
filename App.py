import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import time
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Institutional Options Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Institutional Options & Yield Engine")
st.caption("7 DTE Strategy • Black-Scholes Delta Chain Matcher + RSI + ATR + IV Rank + Earnings Guard")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Strategy Settings")

weekly_goal = st.sidebar.number_input("Weekly Income Goal ($)", value=2000, step=250)
target_dte = st.sidebar.slider("Target DTE", 7, 30, 7)
target_delta = st.sidebar.slider("Target Delta", 0.10, 0.25, 0.18, 0.01)

watchlist_default = "SNOW, NVDA, TSLA, GOOG, AMD, PLTR, UBER, SPY, QQQ"
user_tickers = st.sidebar.text_area("Watchlist Tickers", value=watchlist_default)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

scan_button = st.sidebar.button("🔄 Scan Market Data", use_container_width=True)

# ==========================================
# DIRECT YAHOO REST API CHART FETCH
# ==========================================
def fetch_chart_rest_api(symbol):
    """Hits Yahoo's direct query API for historical daily candles."""
    end_time = int(datetime.datetime.now().timestamp())
    start_time = int((datetime.datetime.now() - datetime.timedelta(days=1825)).timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_time}&period2={end_time}&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
            
        data = r.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        indicators = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': indicators['open'],
            'High': indicators['high'],
            'Low': indicators['low'],
            'Close': indicators['close'],
            'Volume': indicators['volume']
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        return df.dropna()
    except Exception:
        return None

# ==========================================
# BLACK-SCHOLES DELTA MAPPING ENGINE
# ==========================================
def calculate_black_scholes_delta_strike(current_price, target_delta=0.18, dte=7, iv=0.18, option_type="put"):
    """
    Calculates exact target strike price corresponding to target Delta
    using Black-Scholes inversion for 7 DTE options.
    """
    t_years = max(dte, 1) / 365.0
    r = 0.05  # Risk-free interest rate (~5%)
    
    if option_type == "put":
        d1 = norm.ppf(1.0 - target_delta)
    else:
        d1 = norm.ppf(target_delta)
        
    ln_sk = (d1 * iv * np.sqrt(t_years)) - ((r + 0.5 * (iv ** 2)) * t_years)
    strike = current_price * np.exp(-ln_sk) if option_type == "put" else current_price * np.exp(ln_sk)
    
    return round(strike, 1)

def get_live_option_quote(symbol, option_type="put", target_delta=0.18, iv_estimate=0.18, dte=7):
    """Hits live Yahoo chain and matches the calculated Delta strike directly."""
    try:
        t = yf.Ticker(symbol)
        expirations = t.options
        if not expirations:
            return None, None
            
        current_price = float(t.fast_info.last_price)
        target_date = datetime.datetime.now() + datetime.timedelta(days=dte)
        
        # Find nearest expiration date to target DTE
        best_exp = min(expirations, key=lambda x: abs((datetime.datetime.strptime(x, "%Y-%m-%d") - target_date).days))
        
        # Calculate precise Delta strike mathematically
        calc_strike = calculate_black_scholes_delta_strike(current_price, target_delta=target_delta, dte=dte, iv=iv_estimate, option_type=option_type)
        
        chain = t.option_chain(best_exp)
        df_opts = chain.puts if option_type == "put" else chain.calls
        
        if df_opts.empty:
            return calc_strike, round(current_price * 0.003, 2)

        # Match nearest available exchange strike to calculated delta strike
        df_opts['strike_diff'] = abs(df_opts['strike'] - calc_strike)
        selected = df_opts.sort_values('strike_diff').iloc[0]
        
        bid = float(selected['bid']) if selected['bid'] > 0 else float(selected['lastPrice'])
        ask = float(selected['ask']) if selected['ask'] > 0 else float(selected['lastPrice'])
        mid = (bid + ask) / 2.0 if (bid + ask) > 0 else float(selected['lastPrice'])
        
        return float(selected['strike']), float(mid)
    except Exception:
        return None, None

# ==========================================
# GUARDRAILS (EARNINGS & IV RANK)
# ==========================================
def check_upcoming_earnings(symbol, days_ahead=10):
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if cal is not None and not cal.empty and 'Earnings Date' in cal.index:
            earn_dates = cal.loc['Earnings Date']
            now = datetime.datetime.now().date()
            for d in earn_dates:
                if isinstance(d, (datetime.date, datetime.datetime)):
                    d_date = d.date() if isinstance(d, datetime.datetime) else d
                    days_diff = (d_date - now).days
                    if 0 <= days_diff <= days_ahead:
                        return True, f"Earnings in {days_diff}d"
    except Exception:
        pass
    return False, "Clear"

def compute_iv_rank(df_hist):
    try:
        if df_hist is not None and len(df_hist) >= 30:
            returns = np.log(df_hist['Close'] / df_hist['Close'].shift(1))
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) > 0:
                current_vol = rolling_vol.iloc[-1]
                min_vol = rolling_vol.min()
                max_vol = rolling_vol.max()
                
                if max_vol > min_vol:
                    iv_rank = ((current_vol - min_vol) / (max_vol - min_vol)) * 100
                    return round(iv_rank, 1), float(current_vol)
    except Exception:
        pass
    return 50.0, 0.20

# ==========================================
# PROCESS TICKER
# ==========================================
def process_ticker(ticker, target_dte, target_delta):
    t = yf.Ticker(ticker)
    info = t.fast_info
    
    if not info.last_price or info.last_price <= 0:
        return None

    close = float(info.last_price)
    prev_close = float(info.previous_close)
    daily_change_pct = ((close - prev_close) / prev_close) * 100

    df_hist = fetch_chart_rest_api(ticker)
    
    rsi_val = None
    lower_atr = round(close * 0.95, 2)
    upper_atr = round(close * 1.05, 2)
    iv_rank, current_iv = compute_iv_rank(df_hist)

    if df_hist is not None and len(df_hist) >= 15:
        close_s = df_hist['Close']
        high_s = df_hist['High']
        low_s = df_hist['Low']

        delta_df = close_s.diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        if not pd.isna(rsi_series.iloc[-1]):
            rsi_val = float(rsi_series.iloc[-1])

        high_low = high_s - low_s
        high_close = np.abs(high_s - close_s.shift())
        low_close = np.abs(low_s - close_s.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_s = tr.rolling(14).mean()
        
        last_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else close * 0.02
        weekly_move = last_atr * np.sqrt(5)
        lower_atr = round(close - (1.2 * weekly_move), 2)
        upper_atr = round(close + (1.2 * weekly_move), 2)

    has_earnings, earn_status = check_upcoming_earnings(ticker, days_ahead=10)

    # Determine direction based on daily momentum
    opt_type = "call" if daily_change_pct >= 1.5 else "put"
    
    # Live option chain quote with Black-Scholes Delta mapping
    live_strike, live_mid = get_live_option_quote(
        ticker, 
        option_type=opt_type, 
        target_delta=target_delta, 
        iv_estimate=current_iv if current_iv > 0 else 0.20, 
        dte=target_dte
    )
    
    if live_strike and live_mid and live_mid > 0:
        target_strike = live_strike
        est_midpoint = round(live_mid, 2)
    else:
        target_strike = calculate_black_scholes_delta_strike(close, target_delta=target_delta, dte=target_dte, iv=0.20, option_type=opt_type)
        est_midpoint = round(close * 0.005, 2)

    if has_earnings:
        signal = "⚠️ EARNINGS (WAIT)"
    elif iv_rank < 15.0:
        signal = "⚪ LOW IV (WAIT)"
    elif daily_change_pct <= -1.0 and (rsi_val is None or rsi_val <= 48):
        signal = "🟢 SELL CSP"
    elif daily_change_pct >= 1.5 and (rsi_val is None or rsi_val >= 52):
        signal = "🔴 SELL CC"
    else:
        signal = "⚪ WAIT"

    credit_per_contract = est_midpoint * 100.0

    return {
        "Ticker": ticker,
        "Price": round(close, 2),
        "Signal": signal,
        "IV Rank": f"{iv_rank}%",
        "Earnings": earn_status,
        "Target Strike": target_strike,
        "Mid Premium": est_midpoint,
        "Credit / Contract": f"${credit_per_contract:.2f}",
        "Est. Yield ($)": round(credit_per_contract, 2),
        "Put Wall": lower_atr,   
        "Call Wall": upper_atr,  
        "Max Pain": round(close, 2)
    }

# ==========================================
# SCANNER EXECUTION
# ==========================================
if scan_button or 'scan_data' not in st.session_state:
    with st.spinner("Fetching Live Market & Option Chains via Black-Scholes Matching..."):
        results = []
        failed_tickers = []
        
        for t in tickers:
            res = process_ticker(t, target_dte, target_delta)
            if res:
                results.append(res)
            else:
                failed_tickers.append(t)
            time.sleep(0.1)
            
        st.session_state.scan_data = results
        st.session_state.failed_tickers = failed_tickers

results = st.session_state.get('scan_data', [])
failed = st.session_state.get('failed_tickers', [])

# ==========================================
# RENDER TABLE
# ==========================================
if failed:
    st.warning(f"⚠️ Could not pull Yahoo data for: {', '.join(failed)}.")

if results:
    table_rows = []
    for r in results:
        credit_num = r["Est. Yield ($)"]
        contracts_needed = int(np.ceil(weekly_goal / credit_num)) if credit_num > 0 else 0
        total_collateral = contracts_needed * r["Target Strike"] * 100.0

        table_rows.append({
            "Ticker": r["Ticker"],
            "Price": f"${r['Price']:.2f}",
            "Signal": r["Signal"],
            "IV Rank": r["IV Rank"],
            "Earnings": r["Earnings"],
            "Target Strike": f"${r['Target Strike']:.2f}",
            "Mid Premium": f"${r['Mid Premium']:.2f}",
            "Credit / Contract": r["Credit / Contract"],
            "Contracts": f"{contracts_needed}x",
            "Req. Collateral": f"${total_collateral:,.0f}",
            "Put Wall": f"${r['Put Wall']:.2f}",
            "Call Wall": f"${r['Call Wall']:.2f}",
            "Max Pain": f"${r['Max Pain']:.2f}"
        })

    df_display = pd.DataFrame(table_rows)

    st.subheader("📋 Real-Time Option Chain Table")
    
    def highlight_signal(val):
        if "SELL CSP" in str(val):
            return "background-color: #113824; color: #4EFE96; font-weight: bold;"
        elif "SELL CC" in str(val):
            return "background-color: #4A151B; color: #FF6B6B; font-weight: bold;"
        elif "EARNINGS" in str(val):
            return "background-color: #5C3D00; color: #FFC107; font-weight: bold;"
        return "color: #888888;"

    styled_df = df_display.style.map(highlight_signal, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # ==========================================
    # TECHNICAL ANALYSIS CHART ENGINE
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Technical Confirmation & Volatility Bounds")

    col_select, col_tf = st.columns([2, 1])
    with col_select:
        selected_ticker = st.selectbox("Select Ticker for Setup Verification:", [r["Ticker"] for r in results])
    with col_tf:
        selected_tf = st.selectbox("Chart Timeframe:", ["30 Days", "6 Months", "1 Year", "ALL"], index=2)

    t_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if t_data:
        try:
            with st.spinner(f"Loading Chart for {selected_ticker}..."):
                df_chart_raw = fetch_chart_rest_api(selected_ticker)

            if df_chart_raw is not None and not df_chart_raw.empty:
                tf_days_map = {"30 Days": 30, "6 Months": 180, "1 Year": 365, "ALL": 1825}
                days_to_keep = tf_days_map.get(selected_tf, 365)
                
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
                df_chart = df_chart_raw[df_chart_raw.index >= pd.to_datetime(cutoff_date)].copy()

                if df_chart.empty:
                    df_chart = df_chart_raw.copy()

                close_s = df_chart['Close']
                high_s = df_chart['High']
                low_s = df_chart['Low']
                open_s = df_chart['Open']
                vol_s = df_chart['Volume']

                ema20 = close_s.ewm(span=20, adjust=False).mean()
                ema50 = close_s.ewm(span=50, adjust=False).mean()

                delta_df = close_s.diff()
                gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
                loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_s = 100 - (100 / (1 + rs))
                current_rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

                high_low = high_s - low_s
                high_close = np.abs(high_s - close_s.shift())
                low_close = np.abs(low_s - close_s.shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr_s = tr.rolling(14).mean()
                
                last_price = t_data["Price"]
                last_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else last_price * 0.02
                weekly_move = last_atr * np.sqrt(5)
                lower_atr = round(last_price - (1.2 * weekly_move), 2)
                upper_atr = round(last_price + (1.2 * weekly_move), 2)

                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.06,
                    subplot_titles=(
                        f"<b>{selected_ticker}</b> • Daily Candlesticks & ATR Bounds",
                        "Volume",
                        f"RSI 14 ({round(current_rsi, 1)})"
                    ),
                    row_heights=[0.6, 0.2, 0.2]
                )

                fig.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=open_s, high=high_s,
                    low=low_s, close=close_s,
                    name="Price"
                ), row=1, col=1)

                fig.add_trace(go.Scatter(x=df_chart.index, y=ema20, mode='lines', name='20 EMA', line=dict(color='#00F0FF', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=ema50, mode='lines', name='50 EMA', line=dict(color='#FFD166', width=1.5)), row=1, col=1)

                strike_color = "#4EFE96" if "CSP" in t_data["Signal"] else "#FF6B6B"
                fig.add_hline(
                    y=t_data["Target Strike"], line_dash="dash", line_color=strike_color, line_width=2,
                    annotation_text=f"Target Strike: ${t_data['Target Strike']:.2f}", annotation_position="top right", row=1, col=1
                )

                fig.add_hline(
                    y=lower_atr, line_dash="dot", line_color="#22C55E", opacity=0.7,
                    annotation_text=f"ATR Support: ${lower_atr:.2f}", annotation_position="bottom left", row=1, col=1
                )
                fig.add_hline(
                    y=upper_atr, line_dash="dot", line_color="#EF4444", opacity=0.7,
                    annotation_text=f"ATR Resist: ${upper_atr:.2f}", annotation_position="top left", row=1, col=1
                )

                vol_colors = ['#EF4444' if open_s.iloc[i] > close_s.iloc[i] else '#22C55E' for i in range(len(open_s))]
                fig.add_trace(go.Bar(x=df_chart.index, y=vol_s, name="Volume", marker_color=vol_colors), row=2, col=1)

                fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_s, mode='lines', name='RSI', line=dict(color='#A855F7', width=1.5)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#22C55E", row=3, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    height=720,
                    margin=dict(l=20, r=20, t=80, b=20),
                    xaxis_rangeslider_visible=False,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    config={'scrollZoom': False, 'displayModeBar': True}
                )
            else:
                st.error(f"No chart data returned for {selected_ticker}.")
        except Exception as e:
            st.error(f"Chart Error for {selected_ticker}: {e}")
else:
    st.info("👈 Click '🔄 Scan Market Data' to trigger fresh Yahoo quote updates.")
