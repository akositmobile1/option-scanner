import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"  # Prevents Streamlit filesystem permission errors

import math
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Institutional Options Engine",
    page_icon="⚡",
    layout="wide"
)

# Custom User-Agent Session to prevent Cloud IP Blocking
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# ==========================================
# SAFE INSTITUTIONAL FLOW & WALLS ENGINE
# ==========================================
def analyze_option_flow_and_walls(ticker_obj, close_price, target_dte):
    """Safely calculates Call/Put Walls. If Cloud IP is throttled, uses strike estimates."""
    default_call = round(close_price * 1.05, 2)
    default_put = round(close_price * 0.95, 2)
    default_pain = round(close_price, 2)
    
    try:
        expirations = ticker_obj.options
        if not expirations or len(expirations) == 0:
            return default_call, default_put, default_pain, None

        today = datetime.date.today()
        best_exp = min(expirations, key=lambda x: abs((datetime.datetime.strptime(x, "%Y-%m-%d").date() - today).days - target_dte))

        chain = ticker_obj.option_chain(best_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return default_call, default_put, default_pain, None

        # Call Wall (Max Call Open Interest)
        if 'openInterest' in calls.columns and not calls['openInterest'].isna().all():
            call_wall = float(calls.loc[calls['openInterest'].idxmax()]['strike'])
        else:
            call_wall = default_call

        # Put Wall (Max Put Open Interest)
        if 'openInterest' in puts.columns and not puts['openInterest'].isna().all():
            put_wall = float(puts.loc[puts['openInterest'].idxmax()]['strike'])
        else:
            put_wall = default_put

        # Build OI DataFrame
        calls_df = calls[['strike', 'openInterest']].rename(columns={'openInterest': 'Call_OI'})
        puts_df = puts[['strike', 'openInterest']].rename(columns={'openInterest': 'Put_OI'})
        oi_df = pd.merge(calls_df, puts_df, on='strike', how='outer').fillna(0)
        oi_df = oi_df[(oi_df['strike'] >= close_price * 0.80) & (oi_df['strike'] <= close_price * 1.20)]

        return round(call_wall, 2), round(put_wall, 2), default_pain, oi_df
    except Exception:
        # Returns safe defaults if Yahoo Cloud IP rate limits option chain requests
        return default_call, default_put, default_pain, None

# ==========================================
# SINGLE TICKER PROCESSOR
# ==========================================
def process_single_ticker(ticker, target_dte, target_delta):
    try:
        # Attach custom session with User-Agent header
        data = yf.Ticker(ticker, session=session)
        df = data.history(period="6m")
        
        # Fallback query if standard history fails
        if df.empty:
            df = yf.download(ticker, period="6m", progress=False)

        if df.empty or len(df) < 5:
            return None

        # Handle MultiIndex columns if returned by yf.download
        if isinstance(df.columns, pd.MultiIndex):
            close_series = df['Close'][ticker]
        else:
            close_series = df['Close']

        close = float(close_series.iloc[-1])
        prev_close = float(close_series.iloc[-2]) if len(close_series) >= 2 else close
        daily_change_pct = ((close - prev_close) / prev_close) * 100

        # Technical Indicators
        sma50_series = close_series.rolling(window=50).mean()
        delta_df = close_series.diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50.0
        sma50 = float(sma50_series.iloc[-1]) if not pd.isna(sma50_series.iloc[-1]) else close

        # Institutional Walls Analysis
        call_wall, put_wall, max_pain, oi_df = analyze_option_flow_and_walls(data, close, target_dte)

        # Strategy Signals
        if (rsi < 52 or daily_change_pct <= -1.5) and close >= (sma50 * 0.98):
            signal = "🟢 SELL CSP"
            target_strike = round(close * (1 - (target_delta * 0.18)), 2)
        elif rsi >= 60 and daily_change_pct > -1.5:
            signal = "🔴 SELL CC"
            target_strike = round(close * (1 + (target_delta * 0.18)), 2)
        else:
            signal = "⚪ WAIT"
            target_strike = round(close * 0.92, 2)

        est_midpoint = round(close * 0.012, 2)
        credit_per_contract = est_midpoint * 100.0

        if credit_per_contract >= 350:
            suggested_contracts = 2
        elif credit_per_contract >= 150:
            suggested_contracts = 3
        else:
            suggested_contracts = 1

        est_yield = credit_per_contract * suggested_contracts

        # Format history dataframe for Plotly
        df_clean = pd.DataFrame({"Close": close_series, "SMA50": sma50_series}, index=df.index)

        return {
            "Ticker": ticker,
            "Price": round(close, 2),
            "Signal": signal,
            "Target Strike": target_strike,
            "Mid Premium": est_midpoint,
            "Credit / Contract": f"${credit_per_contract:.2f}",
            "Contracts": suggested_contracts,
            "Est. Yield ($)": round(est_yield, 2),
            "Put Wall": put_wall,
            "Call Wall": call_wall,
            "Max Pain": max_pain,
            "df": df_clean,
            "oi_df": oi_df
        }
    except Exception as e:
        return None

def fetch_all_tickers(ticker_list, target_dte, target_delta):
    results = []
    # Using sequential execution to avoid rate limits on Cloud deployment
    for t in ticker_list:
        res = process_single_ticker(t, target_dte, target_delta)
        if res is not None:
            results.append(res)
    return results

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("⚡ Institutional Options & Yield Engine")
st.caption("7 DTE Strategy • Hedge Fund Open Interest Walls • Technical Price Charts")

st.sidebar.header("Strategy Settings")
weekly_goal = st.sidebar.number_input("Weekly Income Goal ($)", value=2000, step=250)
target_dte = st.sidebar.slider("Target DTE", 7, 30, 7)
target_delta = st.sidebar.slider("Target Delta", 0.10, 0.25, 0.18, 0.01)

watchlist_default = "SNOW, SPCS, NBIS, SKHY, NVDA, TSLA, GOOG, AMD, PLTR, UBER"
user_tickers = st.sidebar.text_area("Watchlist Tickers", value=watchlist_default)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

if 'scan_data' not in st.session_state or st.sidebar.button("🔄 Scan Market & Open Interest"):
    with st.spinner("Analyzing Market Data & Open Interest..."):
        st.session_state.scan_data = fetch_all_tickers(tickers, target_dte, target_delta)

results = st.session_state.scan_data

if results:
    df_display = pd.DataFrame([{
        "Ticker": r["Ticker"],
        "Price": r["Price"],
        "Signal": r["Signal"],
        "Target Strike": r["Target Strike"],
        "Mid Premium": r["Mid Premium"],
        "Credit / Contract": r["Credit / Contract"],
        "Contracts": r["Contracts"],
        "Est. Yield ($)": r["Est. Yield ($)"],
        "Put Wall": r["Put Wall"],
        "Call Wall": r["Call Wall"],
        "Max Pain": r["Max Pain"]
    } for r in results])

    top_trades = df_display[df_display["Signal"].str.contains("SELL")].head(4)
    total_projected = top_trades["Est. Yield ($)"].sum() if not top_trades.empty else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Income", f"${total_projected:,.2f}", f"{((total_projected/weekly_goal)*100):.0f}% of Goal")
    col2.metric("Active Opportunities", f"{len(top_trades)} Trades")
    col3.metric("Scan Time", datetime.datetime.now().strftime("%H:%M EST"))

    st.markdown("---")
    st.subheader("📋 Market Signals & Institutional Open Interest Walls")

    def style_table(val):
        if "SELL CSP" in str(val):
            return "background-color: #113824; color: #4EFE96; font-weight: bold;"
        elif "SELL CC" in str(val):
            return "background-color: #4A151B; color: #FF6B6B; font-weight: bold;"
        return "color: #888888;"

    styled_df = df_display.style.map(style_table, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=380)

    # Interactive Detail Charts
    st.markdown("---")
    st.subheader("📈 Interactive Visual Deep Dive")

    selected_ticker = st.selectbox("Select Ticker for Detailed Charts:", [r["Ticker"] for r in results])
    ticker_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if ticker_data:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown(f"**{selected_ticker} Historical Price vs Target Strike (${ticker_data['Target Strike']})**")
            df_hist = ticker_data["df"]
            
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], mode='lines', name='Price', line=dict(color='#00D2FF', width=2)))
            
            if 'SMA50' in df_hist.columns:
                fig_price.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA50'], mode='lines', name='50 SMA', line=dict(color='#FFD166', width=1, dash='dot')))

            fig_price.add_hline(y=ticker_data['Target Strike'], line_dash="dash", line_color="#4EFE96" if "CSP" in ticker_data['Signal'] else "#FF6B6B",
                                annotation_text=f"Strike: ${ticker_data['Target Strike']}", annotation_position="bottom right")

            fig_price.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_price, use_container_width=True)

        with chart_col2:
            st.markdown(f"**{selected_ticker} Option Open Interest (Call Wall vs Put Wall)**")
            oi_df = ticker_data["oi_df"]

            if oi_df is not None and not oi_df.empty:
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=oi_df['strike'], y=oi_df['Put_OI'], name='Put OI (Support)', marker_color='#4EFE96'))
                fig_oi.add_trace(go.Bar(x=oi_df['strike'], y=oi_df['Call_OI'], name='Call OI (Resistance)', marker_color='#FF6B6B'))

                fig_oi.add_vline(x=ticker_data['Price'], line_dash="solid", line_color="#FFFFFF", annotation_text="Spot Price", annotation_position="top left")

                fig_oi.update_layout(barmode='group', template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Strike Price", yaxis_title="Open Interest (Contracts)")
                st.plotly_chart(fig_oi, use_container_width=True)
            else:
                st.info("Open Interest distribution chart unavailable for this ticker (Rate Limited by Yahoo).")
else:
    st.warning("No market data returned. Click '🔄 Scan Market & Open Interest' in the sidebar.")
