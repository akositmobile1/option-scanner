import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import time
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
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
st.caption("7 DTE Strategy • Multi-Factor Engine (Price + RSI + ATR) • Fast-Info Yahoo Feed")

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
# DIRECT YAHOO REST API CHART FETCH (WITH EPOCH TIME RANGE)
# ==========================================
def fetch_chart_rest_api(symbol):
    """Hits Yahoo's direct query API with explicit epoch range to ensure full historical series."""
    end_time = int(datetime.datetime.now().timestamp())
    start_time = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_time}&period2={end_time}&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
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
# LIGHTWEIGHT YFINANCE FETCH (FAST_INFO)
# ==========================================
def fetch_yahoo_fast(symbol):
    """Uses fast_info to bypass heavy scraping blocks on Cloud IPs."""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        current_price = info.last_price
        prev_close = info.previous_close

        if current_price and current_price > 0:
            return {
                "current": float(current_price),
                "prev_close": float(prev_close)
            }
    except Exception:
        pass
    return None

def process_ticker(ticker, target_dte, target_delta):
    quote = fetch_yahoo_fast(ticker)
    if not quote:
        return None

    close = quote["current"]
    prev_close = quote["prev_close"]
    daily_change_pct = ((close - prev_close) / prev_close) * 100

    # Fetch historical daily data for RSI & ATR calculations
    df_hist = fetch_chart_rest_api(ticker)
    
    rsi_val = 50.0
    lower_atr = round(close * 0.95, 2)
    upper_atr = round(close * 1.05, 2)

    if df_hist is not None and len(df_hist) >= 15:
        close_s = df_hist['Close']
        high_s = df_hist['High']
        low_s = df_hist['Low']

        # 14-period RSI Calculation
        delta_df = close_s.diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        if not pd.isna(rsi_series.iloc[-1]):
            rsi_val = float(rsi_series.iloc[-1])

        # 14-period ATR Calculation & 1.2x Weekly Bounds
        high_low = high_s - low_s
        high_close = np.abs(high_s - close_s.shift())
        low_close = np.abs(low_s - close_s.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_s = tr.rolling(14).mean()
        
        last_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else close * 0.02
        weekly_move = last_atr * np.sqrt(5)
        lower_atr = round(close - (1.2 * weekly_move), 2)
        upper_atr = round(close + (1.2 * weekly_move), 2)

    # Multi-Factor Rules Engine (Price % Change + RSI Confirmation Filter)
    if daily_change_pct <= -1.0 and rsi_val <= 45:
        signal = "🟢 SELL CSP"
        target_strike = round(close * (1 - (target_delta * 0.18)), 2)
    elif daily_change_pct >= 1.5 and rsi_val >= 55:
        signal = "🔴 SELL CC"
        target_strike = round(close * (1 + (target_delta * 0.18)), 2)
    else:
        signal = "⚪ WAIT"
        target_strike = round(close * 0.95, 2)

    # 7 DTE Delta/Yield Formula
    est_midpoint = round(close * 0.012, 2)
    credit_per_contract = est_midpoint * 100.0

    return {
        "Ticker": ticker,
        "Price": round(close, 2),
        "Signal": signal,
        "Target Strike": target_strike,
        "Mid Premium": est_midpoint,
        "Credit / Contract": f"${credit_per_contract:.2f}",
        "Est. Yield ($)": round(credit_per_contract, 2),
        "Put Wall": lower_atr,   # Dynamic ATR support level
        "Call Wall": upper_atr,  # Dynamic ATR resistance level
        "Max Pain": round(close, 2)
    }

# ==========================================
# SCANNER EXECUTION
# ==========================================
if scan_button or 'scan_data' not in st.session_state:
    with st.spinner("Fetching Live Market Quotes via Yahoo Finance..."):
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
# RENDER TABLE (WITH SIZING & MARGIN CALCULATIONS)
# ==========================================
if failed:
    st.warning(f"⚠️ Could not pull Yahoo data for: {', '.join(failed)}. Symbol may be invalid or halted.")

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

    st.subheader("📋 Real-Time Yahoo Market Table")
    
    def highlight_signal(val):
        if "SELL CSP" in str(val):
            return "background-color: #113824; color: #4EFE96; font-weight: bold;"
        elif "SELL CC" in str(val):
            return "background-color: #4A151B; color: #FF6B6B; font-weight: bold;"
        return "color: #888888;"

    styled_df = df_display.style.map(highlight_signal, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # ==========================================
    # TECHNICAL ANALYSIS CHART ENGINE
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Technical Confirmation & Volatility Bounds")

    selected_ticker = st.selectbox("Select Ticker for Detailed Setup Verification:", [r["Ticker"] for r in results])
    t_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if t_data:
        try:
            with st.spinner(f"Loading Chart for {selected_ticker}..."):
                df_chart = fetch_chart_rest_api(selected_ticker)

            if df_chart is not None and not df_chart.empty:
                close_s = df_chart['Close']
                high_s = df_chart['High']
                low_s = df_chart['Low']
                open_s = df_chart['Open']
                vol_s = df_chart['Volume']

                # EMAs
                ema20 = close_s.ewm(span=20, adjust=False).mean()
                ema50 = close_s.ewm(span=50, adjust=False).mean()

                # RSI (14)
                delta_df = close_s.diff()
                gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
                loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_s = 100 - (100 / (1 + rs))
                current_rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

                # ATR (14) & Expected Move Bounds
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

                # Plotly 3-Panel Chart Layout
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=(
                        f"{selected_ticker} Candlesticks, EMAs & Weekly ATR Bounds",
                        "Volume",
                        f"RSI 14 ({round(current_rsi, 1)})"
                    ),
                    row_heights=[0.6, 0.2, 0.2]
                )

                # Panel 1: Candlesticks & EMAs
                fig.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=open_s, high=high_s,
                    low=low_s, close=close_s,
                    name="Price"
                ), row=1, col=1)

                fig.add_trace(go.Scatter(x=df_chart.index, y=ema20, mode='lines', name='20 EMA', line=dict(color='#00F0FF', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=ema50, mode='lines', name='50 EMA', line=dict(color='#FFD166', width=1.5)), row=1, col=1)

                # Target Strike Horizontal Marker
                strike_color = "#4EFE96" if "CSP" in t_data["Signal"] else "#FF6B6B"
                fig.add_hline(
                    y=t_data["Target Strike"], line_dash="dash", line_color=strike_color, line_width=2,
                    annotation_text=f"Target Strike: ${t_data['Target Strike']:.2f}", annotation_position="top right", row=1, col=1
                )

                # ATR Support / Resistance Lines
                fig.add_hline(
                    y=lower_atr, line_dash="dot", line_color="#22C55E", opacity=0.7,
                    annotation_text=f"ATR Support: ${lower_atr:.2f}", annotation_position="bottom left", row=1, col=1
                )
                fig.add_hline(
                    y=upper_atr, line_dash="dot", line_color="#EF4444", opacity=0.7,
                    annotation_text=f"ATR Resist: ${upper_atr:.2f}", annotation_position="top left", row=1, col=1
                )

                # Panel 2: Volume
                vol_colors = ['#EF4444' if open_s.iloc[i] > close_s.iloc[i] else '#22C55E' for i in range(len(open_s))]
                fig.add_trace(go.Bar(x=df_chart.index, y=vol_s, name="Volume", marker_color=vol_colors), row=2, col=1)

                # Panel 3: RSI
                fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_s, mode='lines', name='RSI', line=dict(color='#A855F7', width=1.5)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#22C55E", row=3, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    height=680,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"No chart data returned for {selected_ticker}.")
        except Exception as e:
            st.error(f"Chart Error for {selected_ticker}: {e}")
else:
    st.info("👈 Click '🔄 Scan Market Data' to trigger fresh Yahoo quote updates.")
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import time
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
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
st.caption("7 DTE Strategy • Multi-Factor Engine (Price + RSI + ATR) • Fast-Info Yahoo Feed")

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
# DIRECT YAHOO REST API CHART FETCH (WITH EPOCH TIME RANGE)
# ==========================================
def fetch_chart_rest_api(symbol):
    """Hits Yahoo's direct query API with explicit epoch range to ensure full historical series."""
    end_time = int(datetime.datetime.now().timestamp())
    start_time = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_time}&period2={end_time}&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
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
# LIGHTWEIGHT YFINANCE FETCH (FAST_INFO)
# ==========================================
def fetch_yahoo_fast(symbol):
    """Uses fast_info to bypass heavy scraping blocks on Cloud IPs."""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        current_price = info.last_price
        prev_close = info.previous_close

        if current_price and current_price > 0:
            return {
                "current": float(current_price),
                "prev_close": float(prev_close)
            }
    except Exception:
        pass
    return None

def process_ticker(ticker, target_dte, target_delta):
    quote = fetch_yahoo_fast(ticker)
    if not quote:
        return None

    close = quote["current"]
    prev_close = quote["prev_close"]
    daily_change_pct = ((close - prev_close) / prev_close) * 100

    # Fetch historical daily data for RSI & ATR calculations
    df_hist = fetch_chart_rest_api(ticker)
    
    rsi_val = 50.0
    lower_atr = round(close * 0.95, 2)
    upper_atr = round(close * 1.05, 2)

    if df_hist is not None and len(df_hist) >= 15:
        close_s = df_hist['Close']
        high_s = df_hist['High']
        low_s = df_hist['Low']

        # 14-period RSI Calculation
        delta_df = close_s.diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        if not pd.isna(rsi_series.iloc[-1]):
            rsi_val = float(rsi_series.iloc[-1])

        # 14-period ATR Calculation & 1.2x Weekly Bounds
        high_low = high_s - low_s
        high_close = np.abs(high_s - close_s.shift())
        low_close = np.abs(low_s - close_s.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_s = tr.rolling(14).mean()
        
        last_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else close * 0.02
        weekly_move = last_atr * np.sqrt(5)
        lower_atr = round(close - (1.2 * weekly_move), 2)
        upper_atr = round(close + (1.2 * weekly_move), 2)

    # Multi-Factor Rules Engine (Price % Change + RSI Confirmation Filter)
    if daily_change_pct <= -1.0 and rsi_val <= 45:
        signal = "🟢 SELL CSP"
        target_strike = round(close * (1 - (target_delta * 0.18)), 2)
    elif daily_change_pct >= 1.5 and rsi_val >= 55:
        signal = "🔴 SELL CC"
        target_strike = round(close * (1 + (target_delta * 0.18)), 2)
    else:
        signal = "⚪ WAIT"
        target_strike = round(close * 0.95, 2)

    # 7 DTE Delta/Yield Formula
    est_midpoint = round(close * 0.012, 2)
    credit_per_contract = est_midpoint * 100.0

    return {
        "Ticker": ticker,
        "Price": round(close, 2),
        "Signal": signal,
        "Target Strike": target_strike,
        "Mid Premium": est_midpoint,
        "Credit / Contract": f"${credit_per_contract:.2f}",
        "Est. Yield ($)": round(credit_per_contract, 2),
        "Put Wall": lower_atr,   # Dynamic ATR support level
        "Call Wall": upper_atr,  # Dynamic ATR resistance level
        "Max Pain": round(close, 2)
    }

# ==========================================
# SCANNER EXECUTION
# ==========================================
if scan_button or 'scan_data' not in st.session_state:
    with st.spinner("Fetching Live Market Quotes via Yahoo Finance..."):
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
# RENDER TABLE (WITH SIZING & MARGIN CALCULATIONS)
# ==========================================
if failed:
    st.warning(f"⚠️ Could not pull Yahoo data for: {', '.join(failed)}. Symbol may be invalid or halted.")

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

    st.subheader("📋 Real-Time Yahoo Market Table")
    
    def highlight_signal(val):
        if "SELL CSP" in str(val):
            return "background-color: #113824; color: #4EFE96; font-weight: bold;"
        elif "SELL CC" in str(val):
            return "background-color: #4A151B; color: #FF6B6B; font-weight: bold;"
        return "color: #888888;"

    styled_df = df_display.style.map(highlight_signal, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # ==========================================
    # TECHNICAL ANALYSIS CHART ENGINE
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Technical Confirmation & Volatility Bounds")

    selected_ticker = st.selectbox("Select Ticker for Detailed Setup Verification:", [r["Ticker"] for r in results])
    t_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if t_data:
        try:
            with st.spinner(f"Loading Chart for {selected_ticker}..."):
                df_chart = fetch_chart_rest_api(selected_ticker)

            if df_chart is not None and not df_chart.empty:
                close_s = df_chart['Close']
                high_s = df_chart['High']
                low_s = df_chart['Low']
                open_s = df_chart['Open']
                vol_s = df_chart['Volume']

                # EMAs
                ema20 = close_s.ewm(span=20, adjust=False).mean()
                ema50 = close_s.ewm(span=50, adjust=False).mean()

                # RSI (14)
                delta_df = close_s.diff()
                gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
                loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_s = 100 - (100 / (1 + rs))
                current_rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

                # ATR (14) & Expected Move Bounds
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

                # Plotly 3-Panel Chart Layout
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=(
                        f"{selected_ticker} Candlesticks, EMAs & Weekly ATR Bounds",
                        "Volume",
                        f"RSI 14 ({round(current_rsi, 1)})"
                    ),
                    row_heights=[0.6, 0.2, 0.2]
                )

                # Panel 1: Candlesticks & EMAs
                fig.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=open_s, high=high_s,
                    low=low_s, close=close_s,
                    name="Price"
                ), row=1, col=1)

                fig.add_trace(go.Scatter(x=df_chart.index, y=ema20, mode='lines', name='20 EMA', line=dict(color='#00F0FF', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=ema50, mode='lines', name='50 EMA', line=dict(color='#FFD166', width=1.5)), row=1, col=1)

                # Target Strike Horizontal Marker
                strike_color = "#4EFE96" if "CSP" in t_data["Signal"] else "#FF6B6B"
                fig.add_hline(
                    y=t_data["Target Strike"], line_dash="dash", line_color=strike_color, line_width=2,
                    annotation_text=f"Target Strike: ${t_data['Target Strike']:.2f}", annotation_position="top right", row=1, col=1
                )

                # ATR Support / Resistance Lines
                fig.add_hline(
                    y=lower_atr, line_dash="dot", line_color="#22C55E", opacity=0.7,
                    annotation_text=f"ATR Support: ${lower_atr:.2f}", annotation_position="bottom left", row=1, col=1
                )
                fig.add_hline(
                    y=upper_atr, line_dash="dot", line_color="#EF4444", opacity=0.7,
                    annotation_text=f"ATR Resist: ${upper_atr:.2f}", annotation_position="top left", row=1, col=1
                )

                # Panel 2: Volume
                vol_colors = ['#EF4444' if open_s.iloc[i] > close_s.iloc[i] else '#22C55E' for i in range(len(open_s))]
                fig.add_trace(go.Bar(x=df_chart.index, y=vol_s, name="Volume", marker_color=vol_colors), row=2, col=1)

                # Panel 3: RSI
                fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_s, mode='lines', name='RSI', line=dict(color='#A855F7', width=1.5)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#22C55E", row=3, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    height=680,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"No chart data returned for {selected_ticker}.")
        except Exception as e:
            st.error(f"Chart Error for {selected_ticker}: {e}")
else:
    st.info("👈 Click '🔄 Scan Market Data' to trigger fresh Yahoo quote updates.")
