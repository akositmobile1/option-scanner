import math
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="High-Yield Options & Institutional Flow Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BLACK-SCHOLES GREEK ENGINE
# ==========================================
def calculate_greeks(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    cdf_d1 = norm.cdf(d1)
    
    if option_type.lower() == "call":
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1.0
        
    return round(float(delta), 3)

# ==========================================
# INSTITUTIONAL FLOW & GEX ENGINE
# ==========================================
def analyze_option_flow_and_walls(ticker_obj, close_price, target_dte):
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None, None, None, None

        today = datetime.date.today()
        best_exp = min(expirations, key=lambda x: abs((datetime.datetime.strptime(x, "%Y-%m-%d").date() - today).days - target_dte))

        chain = ticker_obj.option_chain(best_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return None, None, None, None

        # Call Wall (Strike with Max Call Open Interest)
        call_wall = float(calls.loc[calls['openInterest'].idxmax()]['strike']) if not calls['openInterest'].isna().all() else close_price * 1.05
        
        # Put Wall (Strike with Max Put Open Interest)
        put_wall = float(puts.loc[puts['openInterest'].idxmax()]['strike']) if not puts['openInterest'].isna().all() else close_price * 0.95

        # Max Pain Calculation
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        total_loss = {}
        
        for s in strikes:
            call_loss = calls[calls['strike'] < s].apply(lambda row: (s - row['strike']) * row['openInterest'], axis=1).sum()
            put_loss = puts[puts['strike'] > s].apply(lambda row: (row['strike'] - s) * row['openInterest'], axis=1).sum()
            total_loss[s] = call_loss + put_loss

        max_pain = min(total_loss, key=total_loss.get) if total_loss else close_price

        # Merge for visual distribution graph
        calls_df = calls[['strike', 'openInterest']].rename(columns={'openInterest': 'Call_OI'})
        puts_df = puts[['strike', 'openInterest']].rename(columns={'openInterest': 'Put_OI'})
        oi_df = pd.merge(calls_df, puts_df, on='strike', how='outer').fillna(0)
        
        # Filter OI chart window around spot price (+/- 25%)
        oi_df = oi_df[(oi_df['strike'] >= close_price * 0.75) & (oi_df['strike'] <= close_price * 1.25)]

        return call_wall, put_wall, max_pain, oi_df
    except Exception:
        return None, None, None, None

# ==========================================
# LIVE OPTION CHAIN ENGINE
# ==========================================
def get_live_option_data(ticker_obj, close_price, opt_type, target_dte, target_delta):
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None, None, None

        today = datetime.date.today()
        best_exp = min(expirations, key=lambda x: abs((datetime.datetime.strptime(x, "%Y-%m-%d").date() - today).days - target_dte))

        chain_obj = ticker_obj.option_chain(best_exp)
        options = chain_obj.calls if opt_type == "call" else chain_obj.puts

        if options.empty:
            return None, None, None

        target_strike = close_price * (1 + (target_delta * 0.18)) if opt_type == "call" else close_price * (1 - (target_delta * 0.18))
        idx = (options['strike'] - target_strike).abs().idxmin()
        selected_option = options.loc[idx]

        strike = float(selected_option['strike'])
        bid = float(selected_option.get('bid', 0.0))
        ask = float(selected_option.get('ask', 0.0))
        last_price = float(selected_option.get('lastPrice', 0.0))

        if bid > 0 and ask > 0:
            midpoint = (bid + ask) / 2.0
        elif last_price > 0:
            midpoint = last_price
        else:
            midpoint = 1.20

        return strike, midpoint, best_exp
    except Exception:
        return None, None, None

# ==========================================
# SINGLE TICKER SCANNER PROCESSOR
# ==========================================
def process_single_ticker(ticker, target_dte, target_delta):
    try:
        data = yf.Ticker(ticker)
        df = data.history(period="6m")
        
        if len(df) < 15:
            return None

        close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
        daily_change_pct = ((close - prev_close) / prev_close) * 100

        # Technical Indicators
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        delta_df = df['Close'].diff()
        gain = (delta_df.where(delta_df > 0, 0)).rolling(window=14).mean()
        loss = (-delta_df.where(delta_df < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])
        sma50 = float(df['SMA50'].iloc[-1]) if not pd.isna(df['SMA50'].iloc[-1]) else close

        # Institutional Flow Walls
        call_wall, put_wall, max_pain, oi_df = analyze_option_flow_and_walls(data, close, target_dte)

        # Signal Logic
        if (rsi < 52 or daily_change_pct <= -1.5) and close >= (sma50 * 0.98):
            signal = "🟢 SELL CSP"
            opt_type = "put"
        elif rsi >= 60 and daily_change_pct > -1.5:
            signal = "🔴 SELL CC"
            opt_type = "call"
        else:
            signal = "⚪ WAIT"
            opt_type = "put"

        # Option Details
        live_strike, live_midpoint, best_exp = get_live_option_data(data, close, opt_type, target_dte, target_delta)
        if live_strike is None:
            live_strike = close * (0.92 if opt_type == "put" else 1.08)
            live_midpoint = 1.50
            best_exp = f"{target_dte} DTE"

        premium_per_contract = live_midpoint * 100.0

        # Contract Allocation Logic
        if premium_per_contract >= 350:
            suggested_contracts = 2
        elif premium_per_contract >= 150:
            suggested_contracts = 3
        else:
            suggested_contracts = 1

        est_trade_yield = premium_per_contract * suggested_contracts

        return {
            "Ticker": ticker,
            "Price": round(close, 2),
            "Signal": signal,
            "Target Strike": round(live_strike, 2),
            "Mid Premium": round(live_midpoint, 2),
            "Credit / Contract": f"${premium_per_contract:.2f}",
            "Contracts": suggested_contracts,
            "Est. Yield ($)": round(est_trade_yield, 2),
            "Put Wall (Support)": round(put_wall, 2) if put_wall else "N/A",
            "Call Wall (Resist)": round(call_wall, 2) if call_wall else "N/A",
            "Max Pain": round(max_pain, 2) if max_pain else "N/A",
            "Expiration": best_exp,
            "df": df,
            "oi_df": oi_df
        }
    except Exception:
        return None

def fetch_all_tickers(ticker_list, target_dte, target_delta):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_ticker, t, target_dte, target_delta) for t in ticker_list]
        for f in futures:
            res = f.result()
            if res:
                results.append(res)
    return results

# ==========================================
# DASHBOARD INTERFACE
# ==========================================
st.title("⚡ Institutional Options & Yield Engine")
st.caption("7 DTE Strategy • Hedge Fund Open Interest Walls • Technical Price Charts")

# Sidebar
st.sidebar.header("Strategy Settings")
weekly_goal = st.sidebar.number_input("Weekly Goal ($)", value=2000, step=250)
target_dte = st.sidebar.slider("Target DTE", min_value=7, max_value=30, value=7)
target_delta = st.sidebar.slider("Target Delta", min_value=0.10, max_value=0.25, value=0.18, step=0.01)

watchlist_default = "SNOW, SPCS, NBIS, SKHY, NVDA, TSLA, GOOG, AMD, PLTR, UBER"
user_tickers = st.sidebar.text_area("Watchlist Tickers", value=watchlist_default)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

if 'scan_data' not in st.session_state or st.sidebar.button("🔄 Scan Market & Open Interest"):
    with st.spinner("Analyzing Option Chains, Institutional Open Interest & Price Action..."):
        st.session_state.scan_data = fetch_all_tickers(tickers, target_dte, target_delta)

results = st.session_state.scan_data

if results:
    # Convert to Display DataFrame
    display_data = []
    for r in results:
        display_data.append({
            "Ticker": r["Ticker"],
            "Price": r["Price"],
            "Signal": r["Signal"],
            "Target Strike": r["Target Strike"],
            "Mid Premium": r["Mid Premium"],
            "Credit / Contract": r["Credit / Contract"],
            "Contracts": r["Contracts"],
            "Est. Yield ($)": r["Est. Yield ($)"],
            "Put Wall": r["Put Wall (Support)"],
            "Call Wall": r["Call Wall (Resist)"],
            "Max Pain": r["Max Pain"],
            "Expiration": r["Expiration"]
        })
    
    df_res = pd.DataFrame(display_data)

    # Portfolio Metrics
    top_trades = df_res[df_res["Signal"].str.contains("SELL")].head(4)
    total_projected = top_trades["Est. Yield ($)"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Income", f"${total_projected:,.2f}", f"{((total_projected/weekly_goal)*100):.0f}% of $2k Goal")
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

    styled_df = df_res.style.map(style_table, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=380)

    # ==========================================
    # VISUAL CHARTS & DEEP DIVE SECTION
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Interactive Visual Deep Dive")

    selected_ticker = st.selectbox("Select Ticker to View Price Action & Institutional Open Interest:", [r["Ticker"] for r in results])
    ticker_data = next((item for item in results if item["Ticker"] == selected_ticker), None)

    if ticker_data:
        chart_col1, chart_col2 = st.columns(2)

        # CHART 1: Technical Price Chart with Strike Line
        with chart_col1:
            st.markdown(f"**{selected_ticker} Price Action vs Target Strike (${ticker_data['Target Strike']})**")
            df_hist = ticker_data["df"]
            
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], mode='lines', name='Price', line=dict(color='#00D2FF', width=2)))
            
            if 'SMA50' in df_hist.columns:
                fig_price.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA50'], mode='lines', name='50 SMA', line=dict(color='#FFD166', width=1, dash='dot')))

            # Add Target Strike Line
            fig_price.add_hline(y=ticker_data['Target Strike'], line_dash="dash", line_color="#4EFE96" if "CSP" in ticker_data['Signal'] else "#FF6B6B",
                                annotation_text=f"Target Strike: ${ticker_data['Target Strike']}", annotation_position="bottom right")

            fig_price.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_price, use_container_width=True)

        # CHART 2: Open Interest Distribution (Call Wall vs Put Wall)
        with chart_col2:
            st.markdown(f"**{selected_ticker} Option Open Interest (Call Wall vs Put Wall)**")
            oi_df = ticker_data["oi_df"]

            if oi_df is not None and not oi_df.empty:
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=oi_df['strike'], y=oi_df['Put_OI'], name='Put OI (Support)', marker_color='#4EFE96'))
                fig_oi.add_trace(go.Bar(x=oi_df['strike'], y=oi_df['Call_OI'], name='Call OI (Resistance)', marker_color='#FF6B6B'))

                # Highlight Spot Price
                fig_oi.add_vline(x=ticker_data['Price'], line_dash="solid", line_color="#FFFFFF", annotation_text="Spot Price", annotation_position="top left")

                fig_oi.update_layout(barmode='group', template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Strike Price", yaxis_title="Open Interest (Contracts)")
                st.plotly_chart(fig_oi, use_container_width=True)
            else:
                st.info("Open Interest distribution chart not available for this ticker.")
else:
    st.error("Error loading market data.")
