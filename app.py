"""Configurable weekly RSI crossing dashboard. Run: streamlit run app.py"""
from datetime import date, timedelta
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Weekly RSI Crossings", page_icon="📈", layout="wide")
st.title("Weekly RSI crossings")
st.caption("Find weekly RSI crossings for your chosen ticker, period, average, and level.")

with st.sidebar:
    st.header("Backtest settings")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    years = st.number_input("Backtest length (years)", min_value=1, max_value=40, value=10, step=1)
    period = st.number_input("RSI period (weeks)", min_value=2, max_value=100, value=14, step=1)
    average = st.selectbox("RSI average type", ["Wilder's", "Simple", "Exponential", "Weighted", "Hull"])
    trigger = st.number_input("Crossing level", min_value=1, max_value=99, value=36, step=1)
    overbought = st.number_input("Overbought reference", min_value=1, max_value=99, value=70, step=1)
    st.caption("Only completed weeks are shown. The signal is known after the week's final trading session.")

if not ticker or not re.fullmatch(r"[A-Z0-9.^=\-]{1,25}", ticker):
    st.error("Enter one valid ticker, such as AAPL, BRK-B, or ^GSPC.")
    st.stop()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True,
                       progress=False, threads=False, multi_level_index=False, timeout=20)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def weekly_closes(daily: pd.DataFrame, today: date) -> pd.DataFrame:
    close = daily["Close"].dropna().sort_index()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    frame = pd.DataFrame({"Close": close, "Trading date": close.index})
    frame = frame.resample("W-FRI").last().dropna()
    # A week ending on Friday is complete only after the following Monday starts.
    current_week_end = pd.Timestamp(today).to_period("W-FRI").end_time.normalize()
    return frame.loc[frame.index < current_week_end].copy()


def weighted_average(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length, min_periods=length).apply(
        lambda values: np.dot(values, weights) / weights.sum(), raw=True
    )


def hull_average(series: pd.Series, length: int) -> pd.Series:
    half = max(1, round(length / 2))
    root = max(1, round(np.sqrt(length)))
    raw = 2 * weighted_average(series, half) - weighted_average(series, length)
    return weighted_average(raw, root)


def wilders_average(series: pd.Series, length: int) -> pd.Series:
    """Initialize with an SMA of the first length changes, then use alpha=1/length."""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) <= length:
        return pd.Series(result, index=series.index)
    result[length] = np.mean(values[1:length + 1])
    for i in range(length + 1, len(values)):
        result[i] = (result[i - 1] * (length - 1) + values[i]) / length
    return pd.Series(result, index=series.index)


def calculate_rsi(close: pd.Series, length: int, method: str) -> pd.Series:
    delta = close.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    if method == "Wilder's":
        avg_gain, avg_loss = wilders_average(gains, length), wilders_average(losses, length)
    elif method == "Simple":
        avg_gain = gains.rolling(length, min_periods=length).mean()
        avg_loss = losses.rolling(length, min_periods=length).mean()
    elif method == "Exponential":
        avg_gain = gains.ewm(span=length, adjust=False, min_periods=length).mean()
        avg_loss = losses.ewm(span=length, adjust=False, min_periods=length).mean()
    elif method == "Weighted":
        avg_gain, avg_loss = weighted_average(gains, length), weighted_average(losses, length)
    else:
        # Hull's difference of WMAs can dip below zero even for nonnegative gains/losses.
        avg_gain = hull_average(gains, length).clip(lower=0)
        avg_loss = hull_average(losses, length).clip(lower=0)
    total = avg_gain + avg_loss
    rsi = 100 * avg_gain.div(total.where(total != 0))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0) & total.notna(), 50)
    return rsi.rename("RSI")


as_of = date.today()
window_start = pd.Timestamp(as_of) - pd.DateOffset(years=int(years))
fetch_start = window_start - pd.DateOffset(years=2)
try:
    with st.spinner(f"Loading {ticker} prices..."):
        daily = fetch_daily(ticker, fetch_start.date().isoformat(), (as_of + timedelta(days=1)).isoformat())
except Exception as exc:
    st.error(f"Could not load price data for {ticker}. Check the symbol and try again. ({exc})")
    st.stop()
if daily is None or daily.empty or "Close" not in daily:
    st.error(f"No historical prices returned for {ticker}. Check the symbol or try a shorter range.")
    st.stop()

weekly = weekly_closes(daily, as_of)
weekly["RSI"] = calculate_rsi(weekly["Close"], int(period), average)
weekly["Crossed above"] = (weekly["RSI"].shift(1) <= trigger) & (weekly["RSI"] > trigger)
view = weekly.loc[weekly.index >= window_start].copy()
if view.empty or view["RSI"].notna().sum() < 2:
    st.warning("Not enough complete weekly history to calculate RSI for this ticker.")
    st.stop()
signals = view.loc[view["Crossed above"]].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Upward crossings", len(signals))
c2.metric("Latest weekly RSI", f"{view['RSI'].dropna().iloc[-1]:.1f}")
c3.metric("Last completed week", view["Trading date"].iloc[-1].strftime("%b %d, %Y"))

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    row_heights=[0.58, 0.42], subplot_titles=("Adjusted weekly close", f"Weekly {average} RSI ({period})"))
fig.add_trace(go.Scatter(x=view["Trading date"], y=view["Close"], mode="lines",
                         line=dict(color="#3978d5", width=2), name="Adjusted close"), row=1, col=1)
fig.add_trace(go.Scatter(x=view["Trading date"], y=view["RSI"], mode="lines",
                         line=dict(color="#9b63d5", width=2), name=f"RSI ({period})"), row=2, col=1)
for level, color, dash in [(trigger, "#dc6b31", "solid"), (overbought, "#b3a234", "dash")]:
    fig.add_hline(y=level, line_color=color, line_dash=dash, opacity=0.8, row=2, col=1,
                  annotation_text=str(level), annotation_position="right")
if not signals.empty:
    fig.add_trace(go.Scatter(x=signals["Trading date"], y=signals["Close"], mode="markers",
                             marker=dict(symbol="triangle-up", color="#e27038", size=11),
                             name=f"Cross above {trigger}", customdata=signals["RSI"],
                             hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:,.2f}<br>RSI: %{customdata:.2f}<extra>Crossing</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=signals["Trading date"], y=signals["RSI"], mode="markers",
                             marker=dict(symbol="circle", color="#e27038", size=10),
                             name=f"Cross above {trigger}", showlegend=False,
                             hovertemplate="%{x|%b %d, %Y}<br>RSI: %{y:.2f}<extra>Crossing</extra>"), row=2, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig.update_xaxes(title_text="Last trading day of week", row=2, col=1)
fig.update_layout(height=680, hovermode="x unified", margin=dict(l=30, r=35, t=65, b=35),
                  legend=dict(orientation="h", y=1.08), template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Crossing dates")
if signals.empty:
    st.info(f"No upward crossings of {trigger} in this backtest window.")
else:
    table = pd.DataFrame({"Date": signals["Trading date"].dt.strftime("%Y-%m-%d"),
                          "Weekly close": signals["Close"].round(2),
                          "Prior RSI": weekly["RSI"].shift(1).loc[signals.index].round(2),
                          "RSI at cross": signals["RSI"].round(2)})
    table = table.iloc[::-1].reset_index(drop=True)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button("Download crossing dates (CSV)", table.to_csv(index=False),
                       file_name=f"{ticker.replace('^', '')}_{average.lower().replace(' ', '_').replace(chr(39), '')}_rsi{period}_above{trigger}.csv", mime="text/csv")
st.caption("Weekly closes use the last available trading session in each Friday-ending week. Prices are split and dividend adjusted. "
           "Wilder uses a simple-average seed then 1/period smoothing. Exponential uses EMA span=period. "
           "Weighted gives more weight to recent changes. Hull uses HMA of gains and losses, clipped at zero. "
           "Two years of extra history warm up RSI before the selected backtest window. "
           f"A cross requires previous weekly RSI ≤ {trigger} and current weekly RSI > {trigger}; the first visible week can be a signal.")
