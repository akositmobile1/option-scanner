import math
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Options Alpha Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# BLACK-SCHOLES & GREEKS ENGINE
# ==========================================
def black_scholes_greeks(S, K, T, r, sigma, option_type="put"):
    """
    Calculates Black-Scholes price and Greeks (Delta, Theta, Vega).
    S: Spot Price, K: Strike Price, T: Time to Expiration in Years
    r: Risk-Free Rate, sigma: Implied Volatility
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365.0
    else:  # Put
        delta = norm.cdf(d1) - 1.0
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365.0

    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100.0  # 1% move impact
    return {"delta": delta, "theta": theta, "vega": vega}


# ==========================================
# DATA FETCHING & PROCESSING ENGINE
# ==========================================
@st.cache_data(ttl=300)
def fetch_stock_data(ticker_symbol):
    """Fetches underlying equity metadata, history, and earnings dates."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1y")
        if history.empty:
            return None

        spot_price = history["Close"].iloc[-1]

        # Calculate Historical Volatility (HV20)
        log_returns = np.log(history["Close"] / history["Close"].shift(1))
        hv20 = np.std(log_returns.tail(20)) * np.sqrt(252)

        # Retrieve next earnings date if available
        next_earnings = "N/A"
        try:
            calendar = ticker.calendar
            if calendar is not None and "Earnings Date" in calendar:
                next_earnings = calendar["Earnings Date"][0].strftime("%Y-%m-%d")
            elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
                next_earnings = pd.to_datetime(calendar.iloc[0, 0]).strftime("%Y-%m-%d")
        except Exception:
            pass

        return {
            "ticker": ticker,
            "spot_price": spot_price,
            "history": history,
            "hv20": hv20,
            "next_earnings": next_earnings,
            "expirations": ticker.expirations,
        }
    except Exception:
        return None


def process_option_chain(
    ticker_data, target_dte_min, target_dte_max, target_delta_max, strategy_type="CSP"
):
    """Processes option chains across target DTE windows and filters contracts."""
    ticker = ticker_data["ticker"]
    spot_price = ticker_data["spot_price"]
    expirations = ticker_data["expirations"]

    today = datetime.date.today()
    opportunities = []

    for exp_str in expirations:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        if target_dte_min <= dte <= target_dte_max:
            try:
                opt_chain = ticker.option_chain(exp_str)
                chain = opt_chain.puts if strategy_type == "CSP" else opt_chain.calls
            except Exception:
                continue

            T = max(dte, 1) / 365.0
            risk_free_rate = 0.045  # Assumed 4.5% Treasury baseline

            for _, row in chain.iterrows():
                strike = row["strike"]
                bid = row["bid"]
                ask = row["ask"]
                iv = row["impliedVolatility"]
                volume = row["volume"]
                open_interest = row["openInterest"]

                # Skip illiquid or zero-bid options
                if bid <= 0.05 or iv <= 0.01:
                    continue

                mid_price = (bid + ask) / 2.0

                # Compute Greeks
                greeks = black_scholes_greeks(
                    spot_price,
                    strike,
                    T,
                    risk_free_rate,
                    iv,
                    option_type="put" if strategy_type == "CSP" else "call",
                )
                abs_delta = abs(greeks["delta"])

                # Filter contracts by strategy criteria
                if strategy_type == "CSP":
                    # OTM Puts only
                    if strike >= spot_price or abs_delta > target_delta_max:
                        continue
                    capital_required = strike * 100
                    return_on_capital = (mid_price * 100) / capital_required
                    ann_return = return_on_capital * (365 / dte)
                    buffer_pct = ((spot_price - strike) / spot_price) * 100

                elif strategy_type == "CC":
                    # OTM Calls only
                    if strike <= spot_price or abs_delta > target_delta_max:
                        continue
                    capital_required = spot_price * 100
                    return_on_capital = (mid_price * 100) / capital_required
                    ann_return = return_on_capital * (365 / dte)
                    buffer_pct = ((strike - spot_price) / spot_price) * 100

                opportunities.append(
                    {
                        "Ticker": ticker_data["ticker"].ticker,
                        "Spot Price": round(spot_price, 2),
                        "Expiration": exp_str,
                        "DTE": dte,
                        "Strike": strike,
                        "Buffer (%)": round(buffer_pct, 2),
                        "Bid": bid,
                        "Ask": ask,
                        "Mid Premium": round(mid_price, 2),
                        "IV (%)": round(iv * 100, 1),
                        "Delta": round(greeks["delta"], 3),
                        "Theta": round(greeks["theta"], 3),
                        "ROC (%)": round(return_on_capital * 100, 2),
                        "Ann ROC (%)": round(ann_return * 100, 2),
                        "Volume": int(volume) if pd.notnull(volume) else 0,
                        "Open Int": int(open_interest) if pd.notnull(open_interest) else 0,
                        "Next Earnings": ticker_data["next_earnings"],
                    }
                )

    return pd.DataFrame(opportunities)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("⚙️ Strategy Parameters")

tickers_input = st.sidebar.text_input(
    "Watchlist Tickers (comma-separated)",
    value="TMUS, TSLA, NVDA, AMD, PLTR, UBER, SOFI",
)

strategy = st.sidebar.radio(
    "Option Strategy",
    options=["Cash-Secured Put (CSP)", "Covered Call (CC)"],
    index=0,
)
strategy_code = "CSP" if strategy == "Cash-Secured Put (CSP)" else "CC"

st.sidebar.markdown("---")
st.sidebar.subheader("Expiration & Delta Controls")

col_dte1, col_dte2 = st.sidebar.columns(2)
with col_dte1:
    min_dte = st.number_input("Min DTE", min_value=1, max_value=180, value=20)
with col_dte2:
    max_dte = st.number_input("Max DTE", min_value=1, max_value=360, value=50)

max_delta = st.sidebar.slider(
    "Max Abs Delta",
    min_value=0.05,
    max_value=0.50,
    value=0.30,
    step=0.01,
    help="Target contract risk profile. Delta 0.30 is roughly ~70% probability of expiring OTM.",
)

min_ann_roc = st.sidebar.number_input(
    "Min Annualized ROC (%)", min_value=0.0, max_value=200.0, value=15.0, step=1.0
)

# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================
st.title("📈 High-Yield Options Income Scanner")
st.caption(
    "Scan live option chains for optimal Cash-Secured Puts and Covered Calls based on Delta, DTE, and Annualized Return on Capital."
)

ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.sidebar.button("Run Market Scan", type="primary"):
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker_sym in enumerate(ticker_list):
        status_text.text(f"Fetching option chains for {ticker_sym}...")
        data = fetch_stock_data(ticker_sym)

        if data and data["expirations"]:
            df_opps = process_option_chain(
                data,
                target_dte_min=min_dte,
                target_dte_max=max_dte,
                target_delta_max=max_delta,
                strategy_type=strategy_code,
            )
            if not df_opps.empty:
                all_results.append(df_opps)

        progress_bar.progress((idx + 1) / len(ticker_list))

    status_text.empty()
    progress_bar.empty()

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Apply Minimum Annualized ROC filter
        filtered_df = combined_df[combined_df["Ann ROC (%)"] >= min_ann_roc]
        filtered_df = filtered_df.sort_values(by="Ann ROC (%)", ascending=False)

        # TOP METRICS SUMMARY
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Opportunities Found", len(filtered_df))
        if not filtered_df.empty:
            col_m2.metric("Max Ann. ROC", f"{filtered_df['Ann ROC (%)'].max()}%")
            col_m3.metric("Avg Ann. ROC", f"{round(filtered_df['Ann ROC (%)'].mean(), 1)}%")
            col_m4.metric("Avg Buffer to Strike", f"{round(filtered_df['Buffer (%)'].mean(), 1)}%")

        st.markdown("---")

        # DISPLAY RESULTS TABLE
        st.subheader(f"Filtered Results: {strategy}")

        st.dataframe(
            filtered_df.style.format(
                {
                    "Spot Price": "${:.2f}",
                    "Strike": "${:.2f}",
                    "Bid": "${:.2f}",
                    "Ask": "${:.2f}",
                    "Mid Premium": "${:.2f}",
                    "Buffer (%)": "{:.2f}%",
                    "IV (%)": "{:.1f}%",
                    "Delta": "{:.3f}",
                    "Theta": "{:.3f}",
                    "ROC (%)": "{:.2f}%",
                    "Ann ROC (%)": "{:.2f}%",
                }
            ),
            use_container_width=True,
            height=450,
        )

        # EXPORT DATA
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Scan Results to CSV",
            data=csv_data,
            file_name=f"options_scan_{strategy_code}_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    else:
        st.warning("No option contracts met your criteria across the specified watchlist.")

else:
    st.info("👈 Set your strategy criteria in the sidebar and click **Run Market Scan** to populate opportunities.")
