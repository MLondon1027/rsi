"""Weekly RSI buy and DMI sell dashboard. Run: streamlit run app.py"""
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import re
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

st.set_page_config(page_title="Weekly Trading Signals", page_icon="📈", layout="wide")
st.title("Weekly trading signals")
st.caption("RSI buy crossings and simple DMI sell crossings on completed weekly bars.")

with st.sidebar:
    st.header("Backtest settings")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    years = st.number_input("Backtest length (years)", min_value=1, max_value=40, value=10, step=1)
    period = st.number_input("RSI period (weeks)", min_value=2, max_value=100, value=14, step=1)
    average = st.selectbox("RSI average type", ["Wilder's", "Simple", "Exponential", "Weighted", "Hull"])
    trigger = st.number_input("Crossing level", min_value=0.1, max_value=99.9, value=36.0, step=0.1, format="%.1f")
    overbought = st.number_input("Overbought reference", min_value=0.1, max_value=99.9, value=70.0, step=0.1, format="%.1f")
    st.divider()
    st.header("DMI sell settings")
    dmi_length = st.number_input("DMI and ADX length (weeks)", min_value=2, max_value=100, value=14, step=1)
    adx_cutoff = st.number_input("Low ADX cutoff", min_value=0.0, max_value=100.0,
                                 value=13.5, step=0.1, format="%.1f")
    adx_lookback = st.number_input("ADX comparison lookback (weeks)", min_value=1, max_value=52,
                                   value=5, step=1)
    st.divider()
    st.header("MACD true sell settings")
    macd_fast = st.number_input("MACD fast EMA (weeks)", min_value=1, max_value=100, value=12, step=1)
    macd_slow = st.number_input("MACD slow EMA (weeks)", min_value=int(macd_fast) + 1,
                                max_value=200, value=max(26, int(macd_fast) + 1), step=1)
    macd_signal = st.number_input("MACD signal EMA (weeks)", min_value=1, max_value=100, value=9, step=1)
    st.caption("Only completed weeks are shown. The signal is known after the week's final trading session.")

if not ticker or not re.fullmatch(r"[A-Z0-9.^=\-]{1,25}", ticker):
    st.error("Enter one valid ticker, such as AAPL, BRK-B, or ^GSPC.")
    st.stop()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted daily OHLC from Yahoo's chart endpoint."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
              "period2": int(pd.Timestamp(end, tz="UTC").timestamp()), "interval": "1d"}
    for attempt in range(3):
        response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        if response.status_code in (429, 502, 503) and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        payload = response.json()["chart"]
        if payload.get("error"):
            raise ValueError(str(payload["error"]))
        results = payload.get("result") or []
        if not results:
            return pd.DataFrame(columns=["High", "Low", "Close"])
        item = results[0]
        indicators = item["indicators"]
        raw = indicators["quote"][0]
        adjusted = indicators.get("adjclose") or []
        closes = adjusted[0].get("adjclose") if adjusted else None
        if closes is None:
            closes = raw["close"]
        dates = pd.to_datetime(item["timestamp"], unit="s", utc=True).tz_convert(
            item.get("meta", {}).get("exchangeTimezoneName") or "America/New_York"
        ).tz_localize(None).normalize()
        # Apply the same split/dividend adjustment factor to each day's high and low.
        frame = pd.DataFrame({"Raw close": raw["close"], "High": raw["high"],
                              "Low": raw["low"], "Close": closes}, index=dates)
        factor = frame["Close"].div(frame["Raw close"].replace(0, np.nan))
        frame["High"] *= factor
        frame["Low"] *= factor
        return frame[["High", "Low", "Close"]].dropna().sort_index()
    return pd.DataFrame(columns=["High", "Low", "Close"])


def weekly_closes(daily: pd.DataFrame, today: date) -> pd.DataFrame:
    close = daily["Close"].dropna().sort_index()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    frame = pd.DataFrame({"Close": close, "Trading date": close.index})
    frame = frame.resample("W-FRI").last().dropna()
    # A week ending on Friday is complete only after the following Monday starts.
    current_week_end = pd.Timestamp(today).to_period("W-FRI").end_time.normalize()
    return frame.loc[frame.index < current_week_end].copy()


def weekly_ohlc(daily: pd.DataFrame, today: date) -> pd.DataFrame:
    bars = daily[["High", "Low", "Close"]].dropna().sort_index().copy()
    bars.index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize()
    bars["Trading date"] = bars.index
    bars = bars.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last",
                                         "Trading date": "last"}).dropna()
    current_week_end = pd.Timestamp(today).to_period("W-FRI").end_time.normalize()
    return bars.loc[bars.index < current_week_end].copy()


def simple_dmi(bars: pd.DataFrame, length: int) -> pd.DataFrame:
    """SMA-smoothed directional indicators and SMA-smoothed DX (ADX)."""
    high, low, close = bars["High"], bars["Low"], bars["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    previous_close = close.shift(1)
    tr = pd.concat([high - low, (high - previous_close).abs(),
                    (low - previous_close).abs()], axis=1).max(axis=1)
    # The first week has no prior week, so it cannot enter a directional calculation.
    plus_dm.iloc[0] = minus_dm.iloc[0] = tr.iloc[0] = np.nan
    tr_sum = tr.rolling(length, min_periods=length).sum()
    plus_di = 100 * plus_dm.rolling(length, min_periods=length).sum().div(tr_sum.where(tr_sum != 0))
    minus_di = 100 * minus_dm.rolling(length, min_periods=length).sum().div(tr_sum.where(tr_sum != 0))
    di_total = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs().div(di_total.where(di_total != 0))
    dx = dx.mask((di_total == 0) & di_total.notna(), 0)
    adx = dx.rolling(length, min_periods=length).mean()
    return pd.DataFrame({"DI+": plus_di, "DI-": minus_di, "ADX": adx}, index=bars.index)


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


def calculate_macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    difference = macd - signal_line
    sells = (difference.shift(1) >= 0) & (difference < 0)
    return pd.DataFrame({"MACD": macd, "Signal": signal_line,
                         "MACD - signal": difference, "True sell": sells}, index=close.index)


as_of = pd.Timestamp.now(tz=ZoneInfo("America/New_York")).date()
window_start = pd.Timestamp(as_of) - pd.DateOffset(years=int(years))
fetch_start = window_start - pd.DateOffset(years=3)
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
           "Three years of extra history warm up RSI before the selected backtest window. "
           f"A cross requires previous weekly RSI ≤ {trigger} and current weekly RSI > {trigger}; the first visible week can be a signal.")


st.divider()
st.subheader("Weekly DMI sell signals")
st.caption(f"Simple DMI ({dmi_length}): DI- crosses above DI+. Orange marks ADX < {adx_cutoff:.1f}; "
           f"blue marks ADX higher than {adx_lookback} weeks earlier; purple marks both.")
dmi_bars = weekly_ohlc(daily, as_of)
dmi_bars = dmi_bars.join(simple_dmi(dmi_bars, int(dmi_length)))
dmi_bars["Prior ADX"] = dmi_bars["ADX"].shift(int(adx_lookback))
dmi_bars["Bearish cross"] = ((dmi_bars["DI-"].shift(1) <= dmi_bars["DI+"].shift(1)) &
                            (dmi_bars["DI-"] > dmi_bars["DI+"]))
dmi_bars["Low ADX"] = dmi_bars["Bearish cross"] & (dmi_bars["ADX"] < adx_cutoff)
dmi_bars["Rising ADX"] = dmi_bars["Bearish cross"] & (dmi_bars["ADX"] > dmi_bars["Prior ADX"])
dmi_view = dmi_bars.loc[dmi_bars.index >= window_start].copy()
sell_hits = dmi_view.loc[dmi_view["Low ADX"] | dmi_view["Rising ADX"]].copy()

if dmi_view["ADX"].notna().sum() < 2:
    st.info("This ticker does not have enough weekly OHLC history for the selected DMI length.")
else:
    dfig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                         row_heights=[0.48, 0.29, 0.23],
                         subplot_titles=("Adjusted weekly price and sell signals", "Weekly DI+ and DI-", "Weekly ADX"))
    dfig.add_trace(go.Scatter(x=dmi_view["Trading date"], y=dmi_view["Close"],
                              mode="lines", line=dict(color="#3978d5", width=2),
                              name="Adjusted close"), row=1, col=1)
    dfig.add_trace(go.Scatter(x=dmi_view["Trading date"], y=dmi_view["DI+"], mode="lines",
                              line=dict(color="#2b9a7e", width=2), name="DI+"), row=2, col=1)
    dfig.add_trace(go.Scatter(x=dmi_view["Trading date"], y=dmi_view["DI-"], mode="lines",
                              line=dict(color="#d05b54", width=2), name="DI-"), row=2, col=1)
    dfig.add_trace(go.Scatter(x=dmi_view["Trading date"], y=dmi_view["ADX"], mode="lines",
                              line=dict(color="#8a6eaa", width=2), name="ADX"), row=3, col=1)
    dfig.add_hline(y=adx_cutoff, line_dash="dash", line_color="#e08a4e", row=3, col=1,
                   annotation_text=f"{adx_cutoff:.1f}")
    classes = [
        ("Low ADX only", dmi_view["Low ADX"] & ~dmi_view["Rising ADX"], "#e37735"),
        ("Rising ADX only", dmi_view["Rising ADX"] & ~dmi_view["Low ADX"], "#1683bc"),
        ("Both conditions", dmi_view["Low ADX"] & dmi_view["Rising ADX"], "#894cba"),
    ]
    for label, mask, color in classes:
        hits = dmi_view.loc[mask]
        if hits.empty:
            continue
        dfig.add_trace(go.Scatter(x=hits["Trading date"], y=hits["Close"], mode="markers",
                                  marker=dict(symbol="triangle-down", color=color, size=13),
                                  name=label, customdata=hits[["DI+", "DI-", "ADX", "Prior ADX"]],
                                  hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:,.2f}<br>DI+: %{customdata[0]:.1f}"
                                                "<br>DI-: %{customdata[1]:.1f}<br>ADX: %{customdata[2]:.1f}"
                                                f"<br>ADX {adx_lookback} weeks earlier: %{{customdata[3]:.1f}}<extra>{label}</extra>"),
                       row=1, col=1)
        dfig.add_trace(go.Scatter(x=hits["Trading date"], y=hits["DI-"], mode="markers",
                                  marker=dict(symbol="circle", color=color, size=10),
                                  name=label, showlegend=False), row=2, col=1)
    dfig.update_yaxes(title_text="Price", row=1, col=1)
    dfig.update_yaxes(title_text="DI", row=2, col=1)
    dfig.update_yaxes(title_text="ADX", row=3, col=1)
    dfig.update_xaxes(title_text="Last trading day of week", row=3, col=1)
    dfig.update_layout(height=850, hovermode="x unified", template="plotly_white",
                       margin=dict(l=30, r=35, t=80, b=35), legend=dict(orientation="h", y=1.07))
    st.plotly_chart(dfig, use_container_width=True)

    st.subheader("DMI sell signal dates")
    if sell_hits.empty:
        st.info("No DI- upward crossings met either ADX condition in this backtest window.")
    else:
        sell_table = pd.DataFrame({
            "Date": sell_hits["Trading date"].dt.strftime("%Y-%m-%d"),
            "Weekly close": sell_hits["Close"].round(2),
            "DI+": sell_hits["DI+"].round(1), "DI-": sell_hits["DI-"].round(1),
            "ADX": sell_hits["ADX"].round(1),
            f"ADX {adx_lookback} weeks prior": sell_hits["Prior ADX"].round(1),
            f"ADX < {adx_cutoff:.1f}": sell_hits["Low ADX"],
            f"ADX rising vs {adx_lookback} weeks": sell_hits["Rising ADX"],
        }).iloc[::-1].reset_index(drop=True)
        st.dataframe(sell_table, hide_index=True, use_container_width=True)
        st.download_button("Download DMI sell dates (CSV)", sell_table.to_csv(index=False),
                           file_name=f"{ticker.replace('^', '')}_weekly_dmi_sell_signals.csv", mime="text/csv")
st.caption("Simple DMI uses rolling sums of weekly directional movement and true range for DI+/DI-, "
           "then a simple moving average of DX for ADX. A crossover requires prior DI- ≤ prior DI+ and current DI- > current DI+. "
           "The ADX comparison is with the completed week exactly N weeks earlier. "
           "A sell marker is a rule match, not a guarantee of a price decline.")


st.divider()
st.subheader("Weekly MACD true sells")
st.caption(f"Exponential MACD ({macd_fast}, {macd_slow}, {macd_signal}). A true sell is MACD crossing below its signal line.")
macd_bars = weekly[["Close", "Trading date"]].join(
    calculate_macd(weekly["Close"], int(macd_fast), int(macd_slow), int(macd_signal)))
macd_view = macd_bars.loc[macd_bars.index >= window_start].copy()
macd_hits = macd_view.loc[macd_view["True sell"]]
if macd_view["Signal"].notna().sum() < 2:
    st.info("Not enough complete weekly history for these MACD settings.")
else:
    mfig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         row_heights=[0.55, 0.45], subplot_titles=("Weekly price and true sells", "MACD, signal, and difference"))
    mfig.add_trace(go.Scatter(x=macd_view["Trading date"], y=macd_view["Close"], mode="lines",
                              name="Adjusted close", line=dict(color="#3978d5")), row=1, col=1)
    mfig.add_trace(go.Bar(x=macd_view["Trading date"], y=macd_view["MACD - signal"],
                          name="MACD - signal", marker_color=np.where(macd_view["MACD - signal"] >= 0,
                                                                       "#79b8a7", "#df9191")), row=2, col=1)
    for column, color in [("MACD", "#3978d5"), ("Signal", "#dc963b")]:
        mfig.add_trace(go.Scatter(x=macd_view["Trading date"], y=macd_view[column], mode="lines",
                                  name=column, line=dict(color=color, width=2)), row=2, col=1)
    mfig.add_hline(y=0, line_color="#999999", line_dash="dot", row=2, col=1)
    if not macd_hits.empty:
        mfig.add_trace(go.Scatter(x=macd_hits["Trading date"], y=macd_hits["Close"], mode="markers",
                                  name="True sell", marker=dict(symbol="triangle-down", color="#c62f49", size=13),
                                  customdata=macd_hits[["MACD", "Signal", "MACD - signal"]],
                                  hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:,.2f}<br>MACD: %{customdata[0]:.3f}"
                                                "<br>Signal: %{customdata[1]:.3f}<br>Difference: %{customdata[2]:.3f}<extra>True sell</extra>"),
                       row=1, col=1)
        mfig.add_trace(go.Scatter(x=macd_hits["Trading date"], y=macd_hits["MACD"], mode="markers",
                                  name="True sell", showlegend=False,
                                  marker=dict(color="#c62f49", size=10)), row=2, col=1)
    mfig.update_yaxes(title_text="Price", row=1, col=1)
    mfig.update_yaxes(title_text="MACD", row=2, col=1)
    mfig.update_layout(height=680, template="plotly_white", hovermode="x unified",
                       legend=dict(orientation="h", y=1.08), margin=dict(l=30, r=35, t=70, b=35))
    st.plotly_chart(mfig, use_container_width=True)
    st.subheader("MACD true sell dates")
    if macd_hits.empty:
        st.info("No MACD downward crosses in this backtest window.")
    else:
        macd_table = pd.DataFrame({"Date": macd_hits["Trading date"].dt.strftime("%Y-%m-%d"),
                                   "Weekly close": macd_hits["Close"].round(2),
                                   "MACD": macd_hits["MACD"].round(3),
                                   "Signal": macd_hits["Signal"].round(3),
                                   "MACD - signal": macd_hits["MACD - signal"].round(3),
                                   "True sell": macd_hits["True sell"]}).iloc[::-1].reset_index(drop=True)
        st.dataframe(macd_table, hide_index=True, use_container_width=True)
        st.download_button("Download MACD true sells (CSV)", macd_table.to_csv(index=False),
                           file_name=f"{ticker.replace('^', '')}_weekly_macd_true_sells.csv", mime="text/csv")
st.caption("MACD = fast EMA minus slow EMA. Signal = EMA of MACD. Difference = MACD minus signal. "
           "A downward cross requires the previous difference ≥ 0 and the current difference < 0. "
           "True sell is the name of this rule; it is evaluated independently of the DMI conditions.")


@st.cache_data(ttl=86400, show_spinner=False)
def sp500_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers={"User-Agent": "Weekly-RSI-Research-Dashboard/1.0"}, timeout=25)
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text), match="Symbol")[0]
    if not {"Symbol", "Security"}.issubset(table.columns):
        raise ValueError("Constituent table layout changed")
    result = table[["Symbol", "Security"]].dropna().copy()
    result["Symbol"] = result["Symbol"].str.replace(".", "-", regex=False)
    return result.drop_duplicates("Symbol")


def scan_sp500(members: pd.DataFrame, lookback: int, today: date) -> tuple[pd.DataFrame, int]:
    scan_start = pd.Timestamp(today) - pd.DateOffset(years=3)
    end = (today + timedelta(days=1)).isoformat()
    last_week_end = pd.Timestamp(today).to_period("W-FRI").end_time.normalize()
    first_week_end = last_week_end - pd.Timedelta(weeks=int(lookback))
    rows = []
    skipped = 0
    symbols = members["Symbol"].tolist()
    names = dict(zip(members["Symbol"], members["Security"]))
    progress = st.progress(0, text="Scanning S&P 500 constituents...")
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(fetch_daily, symbol, scan_start.date().isoformat(), end): symbol
                       for symbol in symbols}
            for count, future in enumerate(as_completed(futures), 1):
                symbol = futures[future]
                try:
                    prices = future.result()
                    weeks = weekly_closes(prices, today)
                    if len(weeks) < int(period) + 2:
                        skipped += 1
                        continue
                    weeks["RSI"] = calculate_rsi(weeks["Close"], int(period), average)
                    weeks["Prior RSI"] = weeks["RSI"].shift(1)
                    found = weeks.loc[(weeks.index >= first_week_end) &
                                      (weeks["Prior RSI"] <= trigger) & (weeks["RSI"] > trigger)]
                    for _, hit in found.iterrows():
                        rows.append({"Ticker": symbol, "Company": names[symbol],
                                     "Signal date": hit["Trading date"].strftime("%Y-%m-%d"),
                                     "Prior RSI": round(hit["Prior RSI"], 1),
                                     "RSI at cross": round(hit["RSI"], 1),
                                     "Weekly close": round(hit["Close"], 2)})
                except (requests.RequestException, KeyError, ValueError, TypeError):
                    skipped += 1
                finally:
                    progress.progress(count / len(symbols), text=f"Scanned {count} of {len(symbols)} tickers")
    finally:
        progress.empty()
    result = pd.DataFrame(rows, columns=["Ticker", "Company", "Signal date", "Prior RSI", "RSI at cross", "Weekly close"])
    if not result.empty:
        result = result.sort_values(["Signal date", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return result, skipped


st.divider()
st.subheader("S&P 500 recent signal scanner")
st.caption(f"Uses the current S&P 500 constituent list and your selected {average} RSI ({period}) crossing above {trigger:.1f}.")
lookback_weeks = st.number_input("Completed weeks to search", min_value=1, max_value=52, value=2, step=1)
scan_settings = (int(period), average, float(trigger), int(lookback_weeks))
if st.button("Scan S&P 500", type="primary"):
    try:
        with st.spinner("Loading the current constituent list..."):
            constituents = sp500_constituents()
        scan_results, missing = scan_sp500(constituents, int(lookback_weeks), as_of)
        st.session_state["scan_output"] = (scan_settings, scan_results, len(constituents), missing)
    except Exception as exc:
        st.session_state.pop("scan_output", None)
        st.error(f"The S&P 500 scan could not run. Check the data connection and retry. ({exc})")
if "scan_output" in st.session_state:
    settings, scan_results, total, missing = st.session_state["scan_output"]
    if settings == scan_settings:
        st.write(f"{len(scan_results)} signals across {total - missing} checked ticker symbols; {missing} could not be checked.")
        if total == missing:
            st.error("No ticker data was available. Retry the scan later.")
        elif scan_results.empty:
            st.info("No crossings found in the selected completed weeks among successfully checked tickers.")
        else:
            st.dataframe(scan_results, hide_index=True, use_container_width=True)
            st.download_button("Download scan results (CSV)", scan_results.to_csv(index=False),
                               file_name="sp500_recent_rsi_signals.csv", mime="text/csv")
    else:
        st.info("Settings changed. Run the scan again to update these results.")
st.caption("The scanner uses today's constituents, including for prior weeks. It may take a few minutes. "
           "Recent signals are based on completed Friday-ending weeks; unavailable ticker data are excluded and counted above. "
           "Constituent source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
