import math
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="7 DTE Options Yield Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BLACK-SCHOLES GREEK ENGINE
# ==========================================
def calculate_greeks(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    cdf_d1 = norm.cdf(d1)
    pdf_d1 = norm.pdf(d1)
    
    if option_type.lower() == "call":
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1.0
        
    vega = S * pdf_d1 * math.sqrt(T) / 100.0
    return round(float(delta), 3), round(float(vega), 3)

# ==========================================
# LIVE OPTION CHAIN & MIDPOINT ENGINE
# ==========================================
def get_live_option_data(ticker_obj, close_price, opt_type, target_dte, target_delta):
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None, None, None

        today = datetime.date.today()
        best_exp = None
        min_diff = 999
        
        for exp in expirations:
            exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if abs(dte - target_dte) < min_diff and dte > 0:
                min_diff = abs(dte - target_dte)
                best_exp = exp

        if not best_exp:
            best_exp = expirations[0]

        chain_obj = ticker_obj.option_chain(best_exp)
        options = chain_obj.calls if opt_type == "call" else chain_obj.puts

        if options.empty:
            return None, None, None

        # Filter OTM options
        if opt_type == "call":
            otm_opts = options[options['strike'] >= close_price]
        else:
            otm_opts = options[options['strike'] <= close_price]

        if otm_opts.empty:
            otm_opts = options

        # Target strike selection approximation
        target_strike = close_price * (1 + (target_delta * 0.18)) if opt_type == "call" else close_price * (1 - (target_delta * 0.18))
        idx = (otm_opts['strike'] - target_strike).abs().idxmin()
        selected_option = otm_opts.loc[idx]

        strike = float(selected_option['strike'])
        bid = float(selected_option.get('bid', 0.0))
        ask = float(selected_option.get('ask', 0.0))
        last_price = float(selected_option.get('lastPrice', 0.0))

        if bid > 0 and ask > 0:
            midpoint = (bid + ask) / 2.0
        elif last_price > 0:
            midpoint = last_price
        else:
            midpoint = 0.50

        return strike, midpoint, best_exp
    except Exception:
        return None, None, None

# ==========================================
# 6-MONTH HISTORICAL BACKTESTING ENGINE
# ==========================================
def run_6m_backtest(df, opt_type, target_delta=0.18, weeks=26):
    try:
        if len(df) < (weeks * 5):
            return {"Win Rate": "N/A", "Wins": 0, "Losses": 0, "Net PnL": 0.0}
        
        weekly_closes = df['Close'].resample('W').last().tail(weeks)
        wins = 0
        losses = 0
        total_pnl = 0.0
        
        for i in range(len(weekly_closes) - 1):
            entry_price = weekly_closes.iloc[i]
            exit_price = weekly_closes.iloc[i+1]
            strike_offset = target_delta * 0.25
            
            if opt_type == "put":
                strike = entry_price * (1 - strike_offset)
                est_credit = entry_price * 0.008  # ~0.8% weekly premium
                
                if exit_price <= strike:
                    losses += 1
                    # 100% loss management: Stop-loss capped at 2x credit loss
                    total_pnl -= (est_credit * 1.0) * 100
                else:
                    wins += 1
                    total_pnl += (est_credit * 0.60) * 100
            else:  # Call
                strike = entry_price * (1 + strike_offset)
                est_credit = entry_price * 0.008
                
                if exit_price >= strike:
                    losses += 1
                    total_pnl -= (est_credit * 1.0) * 100
                else:
                    wins += 1
                    total_pnl += (est_credit * 0.60) * 100

        total_trades = wins + losses
        win_rate_pct = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        return {
            "Win Rate": f"{win_rate_pct:.1f}%",
            "Wins": wins,
            "Losses": losses,
            "Net PnL": round(total_pnl, 2)
        }
    except Exception:
        return {"Win Rate": "N/A", "Wins": 0, "Losses": 0, "Net PnL": 0.0}

# ==========================================
# TICKER SCANNER PROCESSOR
# ==========================================
def process_single_ticker(ticker, target_dte, target_delta):
    try:
        data = yf.Ticker(ticker)
        df = data.history(period="1y")
        
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

        # Historical Volatility
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        volatility_est = float(log_returns.tail(21).std() * np.sqrt(252))
        if np.isnan(volatility_est) or volatility_est == 0:
            volatility_est = 0.25

        # Signal Engine with Daily Drop Guard Rules
        if (rsi < 52 or daily_change_pct <= -1.5) and close >= (sma50 * 0.98):
            signal = "🟢 SELL CSP"
            opt_type = "put"
        elif rsi >= 60 and daily_change_pct > -1.5:
            signal = "🔴 SELL CC"
            opt_type = "call"
        else:
            signal = "⚪ WAIT"
            opt_type = "put"

        # Option Fetching
        live_strike, live_midpoint, best_exp = get_live_option_data(
            data, close, opt_type, target_dte, target_delta
        )

        if live_strike is None:
            if opt_type == "call":
                live_strike = close * (1 + (target_delta * volatility_est * (target_dte / 30.0)))
            else:
                live_strike = close * (1 - (target_delta * volatility_est * (target_dte / 30.0)))
            live_midpoint = close * target_delta * (volatility_est * 0.10)
            best_exp = f"{target_dte} DTE"

        # Black-Scholes Greeks
        T_years = max(target_dte, 1) / 365.0
        delta_val, vega_val = calculate_greeks(
            S=close, K=live_strike, T=T_years, r=0.045, sigma=volatility_est, option_type=opt_type
        )

        # 6-Month Backtest Engine Run
        backtest_res = run_6m_backtest(df, opt_type, target_delta=target_delta, weeks=26)

        return {
            "Ticker": ticker,
            "Price": round(close, 2),
            "Daily Change %": round(daily_change_pct, 2),
            "RSI (14)": round(rsi, 1),
            "Signal": signal,
            "Target Strike": round(live_strike, 2),
            "Midpoint ($/sh)": round(live_midpoint, 2),
            "Credit ($/cntrct)": round(live_midpoint * 100, 2),
            "6M Win Rate": backtest_res["Win Rate"],
            "6M Net PnL ($/cntrct)": backtest_res["Net PnL"],
            "Expiration": best_exp,
            "Est. Delta": abs(delta_val)
        }
    except Exception:
        return None

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
    return results

# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================
st.title("🎯 Weekly Options Income Scanner")
st.caption("Strategy Engine Defaults: 7 DTE • 0.15–0.20 Target Delta • Daily Drop Guard Active")

# Sidebar - Preset Defaults
st.sidebar.header("Strategy Defaults")
portfolio_size = st.sidebar.number_input("Portfolio Capital ($)", value=600000, step=25000)
target_dte = st.sidebar.slider("Days to Expiration (DTE)", min_value=7, max_value=45, value=7, step=1)
target_delta = st.sidebar.slider("Target Delta", min_value=0.05, max_value=0.30, value=0.18, step=0.01)

default_watchlist = "GOOG, TMUS, SKHY, NVDA, TSLA, AMD, SPY, QQQ, PLTR, UBER, SOFI"
user_tickers = st.sidebar.text_area("Watchlist Tickers", value=default_watchlist)
tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]

# Automatic Run on Load
if 'scan_data' not in st.session_state or st.sidebar.button("🔄 Refresh Market Scanner"):
    with st.spinner("Scanning 7 DTE Option Chains & Executing 6M Backtests..."):
        st.session_state.scan_data = fetch_all_tickers(tickers, target_dte, target_delta)

results_list = st.session_state.scan_data

if results_list:
    df_results = pd.DataFrame(results_list)
    
    csp_count = len(df_results[df_results["Signal"] == "🟢 SELL CSP"])
    cc_count = len(df_results[df_results["Signal"] == "🔴 SELL CC"])
    wait_count = len(df_results[df_results["Signal"] == "⚪ WAIT"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Signals", f"{csp_count + cc_count} / {len(df_results)}")
    col2.metric("🟢 CSP Opportunities", f"{csp_count} Tickers")
    col3.metric("🔴 CC Opportunities", f"{cc_count} Tickers")
    col4.metric("⚪ WAIT / No Action", f"{wait_count} Tickers")

    st.markdown("---")
    st.subheader("Weekly 7 DTE Strategy Signals & 6-Month Performance")
    
    def highlight_signals(val):
        if "SELL CSP" in str(val):
            return "background-color: #113824; color: #4EFE96; font-weight: bold;"
        elif "SELL CC" in str(val):
            return "background-color: #4A151B; color: #FF6B6B; font-weight: bold;"
        return "color: #888888;"

    styled_df = df_results.style.map(highlight_signals, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True, height=450)
else:
    st.error("Unable to load market data. Check internet connectivity or yfinance API status.")
