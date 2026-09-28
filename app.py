"""Weekly Wilder RSI crossing dashboard. Run: streamlit run app.py"""
from datetime import date, timedelta
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

RSI_PERIOD = 14
OVERBOUGHT = 70
TRIGGER = 36

st.set_page_config(page_title="Weekly RSI Crossings", page_icon="📈", layout="wide")
st.title("Weekly RSI crossings")
st.caption("Wilder RSI (14) · 70 overbought · upward crossing of 36")

with st.sidebar:
    st.header("Backtest settings")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    years = st.number_input("Backtest length (years)", min_value=1, max_value=40, value=10, step=1)
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


def wilders_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Seed Wilder's average with the first 14 changes, then smooth recursively."""
    delta = close.diff()
    gains = delta.clip(lower=0).to_numpy()
    losses = (-delta.clip(upper=0)).to_numpy()
    values = [float("nan")] * len(close)
    if len(close) <= period:
        return pd.Series(values, index=close.index, name="RSI")
    avg_gain = float(pd.Series(gains[1:period + 1]).mean())
    avg_loss = float(pd.Series(losses[1:period + 1]).mean())
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            values[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            values[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return pd.Series(values, index=close.index, name="RSI")


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
weekly["RSI"] = wilders_rsi(weekly["Close"])
weekly["Crossed above 36"] = (weekly["RSI"].shift(1) <= TRIGGER) & (weekly["RSI"] > TRIGGER)
view = weekly.loc[weekly.index >= window_start].copy()
if view.empty or view["RSI"].notna().sum() < 2:
    st.warning("Not enough complete weekly history to calculate RSI for this ticker.")
    st.stop()
signals = view.loc[view["Crossed above 36"]].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Upward crossings", len(signals))
c2.metric("Latest weekly RSI", f"{view['RSI'].dropna().iloc[-1]:.1f}")
c3.metric("Last completed week", view["Trading date"].iloc[-1].strftime("%b %d, %Y"))

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    row_heights=[0.58, 0.42], subplot_titles=("Adjusted weekly close", "Weekly Wilder RSI (14)"))
fig.add_trace(go.Scatter(x=view["Trading date"], y=view["Close"], mode="lines",
                         line=dict(color="#3978d5", width=2), name="Adjusted close"), row=1, col=1)
fig.add_trace(go.Scatter(x=view["Trading date"], y=view["RSI"], mode="lines",
                         line=dict(color="#9b63d5", width=2), name="RSI (14)"), row=2, col=1)
for level, color, dash in [(TRIGGER, "#dc6b31", "solid"), (OVERBOUGHT, "#b3a234", "dash")]:
    fig.add_hline(y=level, line_color=color, line_dash=dash, opacity=0.8, row=2, col=1,
                  annotation_text=str(level), annotation_position="right")
if not signals.empty:
    fig.add_trace(go.Scatter(x=signals["Trading date"], y=signals["Close"], mode="markers",
                             marker=dict(symbol="triangle-up", color="#e27038", size=11),
                             name="Cross above 36", customdata=signals["RSI"],
                             hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:,.2f}<br>RSI: %{customdata:.2f}<extra>Cross above 36</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=signals["Trading date"], y=signals["RSI"], mode="markers",
                             marker=dict(symbol="circle", color="#e27038", size=10),
                             name="Cross above 36", showlegend=False,
                             hovertemplate="%{x|%b %d, %Y}<br>RSI: %{y:.2f}<extra>Cross above 36</extra>"), row=2, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig.update_xaxes(title_text="Last trading day of week", row=2, col=1)
fig.update_layout(height=680, hovermode="x unified", margin=dict(l=30, r=35, t=65, b=35),
                  legend=dict(orientation="h", y=1.08), template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Crossing dates")
if signals.empty:
    st.info("No upward crossings of 36 in this backtest window.")
else:
    table = pd.DataFrame({"Date": signals["Trading date"].dt.strftime("%Y-%m-%d"),
                          "Weekly close": signals["Close"].round(2),
                          "Prior RSI": weekly["RSI"].shift(1).loc[signals.index].round(2),
                          "RSI at cross": signals["RSI"].round(2)})
    table = table.iloc[::-1].reset_index(drop=True)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button("Download crossing dates (CSV)", table.to_csv(index=False),
                       file_name=f"{ticker.replace('^', '')}_weekly_rsi_crossings.csv", mime="text/csv")
st.caption("Weekly closes use the last available trading session in each Friday-ending week. Prices are split and dividend adjusted. "
           "Wilder RSI starts with a 14-week simple average of gains and losses, then uses Wilder smoothing. "
           "Two years of extra history warm up RSI before the selected backtest window. "
           "A cross requires previous weekly RSI ≤ 36 and current weekly RSI > 36; the first visible week can be a signal.")
