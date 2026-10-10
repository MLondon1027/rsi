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
                                   value=4, step=1)
    st.divider()
    st.header("MACD confirmation settings")
    macd_fast = st.number_input("MACD fast EMA (weeks)", min_value=1, max_value=100, value=12, step=1)
    macd_slow = st.number_input("MACD slow EMA (weeks)", min_value=int(macd_fast) + 1,
                                max_value=200, value=max(26, int(macd_fast) + 1), step=1)
    macd_signal = st.number_input("MACD signal EMA (weeks)", min_value=1, max_value=100, value=9, step=1)
    macd_cutoff = st.number_input("True sell: MACD − signal below", value=-1.0,
                                  step=0.1, format="%.3f",
                                  help="Strict cutoff checked on the same DMI sell week. Equality does not qualify.")
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
                              "Low": raw["low"], "Close": closes, "Volume": raw.get("volume", [None] * len(dates))}, index=dates)
        factor = frame["Close"].div(frame["Raw close"].replace(0, np.nan))
        frame["High"] *= factor
        frame["Low"] *= factor
        return frame[["High", "Low", "Close", "Volume"]].dropna(subset=["High", "Low", "Close"]).sort_index()
    return pd.DataFrame(columns=["High", "Low", "Close"])


def weekly_closes(daily: pd.DataFrame, today: date) -> pd.DataFrame:
    return weekly_ohlc(daily, today)


def weekly_ohlc(daily: pd.DataFrame, today: date) -> pd.DataFrame:
    bars = daily[["High", "Low", "Close", "Volume"]].dropna(subset=["High", "Low", "Close"]).sort_index().copy()
    bars.index = pd.DatetimeIndex(bars.index).tz_localize(None).normalize()
    bars["Trading date"] = bars.index
    bars = bars.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last",
                                         "Trading date": "last", "Volume": lambda values: values.sum(min_count=1)}).dropna(subset=["High", "Low", "Close", "Trading date"])
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



def add_volume_check(bars: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    prior = result["Volume"].shift(1).rolling(3, min_periods=3).mean()
    result["Prior 3-week average volume"] = prior
    result["Volume increased?"] = np.where(result["Volume"].isna() | prior.isna(), "Unavailable",
                                            np.where(result["Volume"] > prior, "Yes", "No"))
    return result


def add_sell_signals(bars: pd.DataFrame, length: int, cutoff: float, lookback: int,
                     fast: int, slow: int, signal: int, macd_limit: float) -> pd.DataFrame:
    result = bars.join(simple_dmi(bars, length))
    result["Prior ADX"] = result["ADX"].shift(lookback)
    cross = (result["DI-"].shift(1) <= result["DI+"].shift(1)) & (result["DI-"] > result["DI+"])
    result["Low ADX"] = cross & (result["ADX"] < cutoff)
    result["Rising ADX"] = cross & (result["ADX"] > result["Prior ADX"])
    macd = calculate_macd(result["Close"], fast, slow, signal)
    result["MACD - signal"] = macd["MACD - signal"]
    result["MACD cutoff met"] = result["MACD - signal"] < macd_limit
    result["True sell"] = (result["Low ADX"] | result["Rising ADX"]) & macd["True sell"] & result["MACD cutoff met"]
    return result


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
st.subheader("Buy signals — weekly RSI")
weekly = add_volume_check(weekly)
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

st.subheader("Buy signal dates")
if signals.empty:
    st.info(f"No upward crossings of {trigger} in this backtest window.")
else:
    table = pd.DataFrame({"Date": signals["Trading date"].dt.strftime("%Y-%m-%d"),
                          "Weekly close": signals["Close"].round(2),
                          "Prior RSI": weekly["RSI"].shift(1).loc[signals.index].round(2),
                          "RSI at cross": signals["RSI"].round(2),
                          "Weekly volume": signals["Volume"],
                          "Prior 3-week average volume": signals["Prior 3-week average volume"].round(0),
                          "Volume increased?": signals["Volume increased?"]})
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
st.subheader("Sell signals — weekly DMI with MACD confirmation")
st.caption(f"Simple DMI ({dmi_length}): DI- crosses above DI+. Orange marks ADX < {adx_cutoff:.1f}; "
           f"blue marks ADX higher than {adx_lookback} weeks earlier; purple marks both. "
           f"A red outline marks an existing DMI sell with a same-week MACD downward cross and difference < {macd_cutoff:.3f}.")
dmi_bars = add_sell_signals(weekly_ohlc(daily, as_of), int(dmi_length), float(adx_cutoff),
                            int(adx_lookback), int(macd_fast), int(macd_slow), int(macd_signal), float(macd_cutoff))
dmi_view = dmi_bars.loc[dmi_bars.index >= window_start].copy()
sell_hits = dmi_view.loc[dmi_view["Low ADX"] | dmi_view["Rising ADX"]].copy()

if dmi_view["ADX"].notna().sum() < 2:
    st.info("This ticker does not have enough weekly OHLC history for the selected DMI length.")
else:
    dfig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                         row_heights=[0.40, 0.25, 0.20, 0.15],
                         subplot_titles=("Adjusted weekly price and sell signals", "Weekly DI+ and DI-", "Weekly ADX", "MACD − signal at DMI sell dates only"))
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
                                  name=label, customdata=hits[["DI+", "DI-", "ADX", "Prior ADX", "MACD - signal", "True sell"]],
                                  hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:,.2f}<br>DI+: %{customdata[0]:.1f}"
                                                "<br>DI-: %{customdata[1]:.1f}<br>ADX: %{customdata[2]:.1f}"
                                                f"<br>ADX {adx_lookback} weeks earlier: %{{customdata[3]:.1f}}"
                                                "<br>MACD − signal: %{customdata[4]:.3f}<br>True sell: %{customdata[5]}"
                                                f"<extra>{label}</extra>"),
                       row=1, col=1)
        dfig.add_trace(go.Scatter(x=hits["Trading date"], y=hits["DI-"], mode="markers",
                                  marker=dict(symbol="circle", color=color, size=10),
                                  name=label, showlegend=False), row=2, col=1)
    dfig.add_trace(go.Bar(x=sell_hits["Trading date"], y=sell_hits["MACD - signal"],
                          name="MACD − signal at sell dates", marker_color="#7784a1"), row=4, col=1)
    dfig.add_hline(y=0, line_dash="dot", line_color="#999999", row=4, col=1)
    confirmed = sell_hits.loc[sell_hits["True sell"]]
    if not confirmed.empty:
        dfig.add_trace(go.Scatter(x=confirmed["Trading date"], y=confirmed["Close"], mode="markers",
                                  name="True sell (MACD cross below)",
                                  marker=dict(symbol="circle-open", color="#c62f49", size=22, line=dict(width=2))),
                       row=1, col=1)
    dfig.update_yaxes(title_text="Difference", row=4, col=1)
    dfig.update_yaxes(title_text="Price", row=1, col=1)
    dfig.update_yaxes(title_text="DI", row=2, col=1)
    dfig.update_yaxes(title_text="ADX", row=3, col=1)
    dfig.update_xaxes(title_text="Last trading day of week", row=4, col=1)
    dfig.update_layout(height=1000, hovermode="x unified", template="plotly_white",
                       margin=dict(l=30, r=35, t=80, b=35), legend=dict(orientation="h", y=1.07))
    st.plotly_chart(dfig, use_container_width=True)

    st.subheader("DMI sell signal dates")
    if sell_hits.empty:
        st.info("No DI- upward crossings met either ADX condition in this backtest window.")
    else:
        sell_table = pd.DataFrame({
            "Date": sell_hits["Trading date"].dt.strftime("%Y-%m-%d"),
            "Weekly close": sell_hits["Close"].round(2),
            "MACD - signal": sell_hits["MACD - signal"].round(3),
            f"MACD difference < {macd_cutoff:.3f}": sell_hits["MACD cutoff met"],
            "True sell": sell_hits["True sell"],
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
           "MACD − signal is shown only for these DMI sell dates. True sell requires a same-week MACD downward cross: "
           f"previous difference ≥ 0, current difference < 0, and current difference < the selected cutoff ({macd_cutoff:.3f}).")


@st.cache_data(ttl=86400, show_spinner=False)
def download_membership_history() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/chinobing/historical_sp500_constituents/main/sp_500_historical_components.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_membership_history(pd.read_csv(StringIO(response.text)))


def parse_membership_history(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "tickers"}.issubset(frame.columns):
        raise ValueError("Membership CSV must have date and tickers columns; tickers must be a comma-separated list.")
    result = frame[["date", "tickers"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["members"] = result["tickers"].map(lambda value: frozenset(
        symbol.strip().upper().replace(".", "-") for symbol in str(value).split(",") if symbol.strip()))
    if result.empty or result["members"].map(len).min() < 400:
        raise ValueError("Membership history is empty or contains an incomplete snapshot.")
    return result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def members_on(history: pd.DataFrame, when: pd.Timestamp) -> frozenset:
    position = history["date"].searchsorted(pd.Timestamp(when).normalize(), side="right") - 1
    if position < 0:
        raise ValueError("The selected date precedes membership history coverage.")
    return history.iloc[position]["members"]


def eligible_symbols(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    symbols = set(members_on(history, start))
    for members in history.loc[(history["date"] > start) & (history["date"] <= end), "members"]:
        symbols.update(members)
    return sorted(symbols)


def scan_sp500(history: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
               kind: str, today: date) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    symbols = eligible_symbols(history, start, end)
    price_start = (start - pd.DateOffset(years=3)).date().isoformat()
    price_end = (end + pd.Timedelta(days=1)).date().isoformat()
    rows, errors = [], []
    progress = st.progress(0, text=f"Scanning {kind.lower()} signals...")
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(fetch_daily, symbol, price_start, price_end): symbol for symbol in symbols}
            for count, future in enumerate(as_completed(futures), 1):
                symbol = futures[future]
                try:
                    prices = future.result()
                    if prices.empty:
                        raise ValueError("No price history returned (possibly delisted or renamed)")
                    bars = weekly_ohlc(prices, today)
                    if bars.empty:
                        raise ValueError("No completed weekly price history")
                    if kind == "Buy":
                        bars = add_volume_check(bars)
                        bars["RSI"] = calculate_rsi(bars["Close"], int(period), average)
                        bars["Prior RSI"] = bars["RSI"].shift(1)
                        mask = (bars["Prior RSI"] <= trigger) & (bars["RSI"] > trigger)
                        ready = bars["RSI"].notna()
                    else:
                        bars = add_sell_signals(bars, int(dmi_length), float(adx_cutoff), int(adx_lookback),
                                                int(macd_fast), int(macd_slow), int(macd_signal), float(macd_cutoff))
                        mask = bars["True sell"]
                        ready = bars["ADX"].notna() & bars["MACD - signal"].notna()
                    in_window = (bars["Trading date"] >= start) & (bars["Trading date"] <= end)
                    if not (ready & in_window).any():
                        raise ValueError("Insufficient price history in the selected window")
                    # Filter by membership on each signal date, including intra-year additions/removals.
                    for _, hit in bars.loc[mask & in_window].iterrows():
                        signal_date = hit["Trading date"]
                        if symbol not in members_on(history, signal_date):
                            continue
                        row = {"Ticker": symbol, "Signal date": signal_date.strftime("%Y-%m-%d"),
                               "Weekly close": round(hit["Close"], 2), "S&P 500 member on signal date": "Yes"}
                        if kind == "Buy":
                            row.update({"Prior RSI": round(hit["Prior RSI"], 1), "RSI at cross": round(hit["RSI"], 1),
                                        "Weekly volume": hit["Volume"],
                                        "Prior 3-week average volume": round(hit["Prior 3-week average volume"], 0),
                                        "Volume increased?": hit["Volume increased?"]})
                        else:
                            row.update({"DI+": round(hit["DI+"], 1), "DI-": round(hit["DI-"], 1),
                                        "ADX": round(hit["ADX"], 1),
                                        f"ADX {adx_lookback} weeks prior": round(hit["Prior ADX"], 1),
                                        "MACD - signal": round(hit["MACD - signal"], 3), "True sell": "Yes"})
                        rows.append(row)
                except (requests.RequestException, KeyError, ValueError, TypeError, IndexError) as exc:
                    errors.append({"Ticker": symbol, "Reason": str(exc)[:160]})
                finally:
                    progress.progress(count / len(symbols), text=f"Checked {count} of {len(symbols)} eligible symbols")
    finally:
        progress.empty()
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["Signal date", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return result, pd.DataFrame(errors), len(symbols)


st.divider()
st.subheader("S&P 500 recent signal scanner — Buy and Sell")
st.caption("Uses dated S&P 500 membership on each signal date. Sell scanning returns confirmed True sells using the dashboard settings.")
scan_scope = st.radio("Scan timeframe", ["Recent completed weeks", "Entire selected backtest"], horizontal=True)
lookback_weeks = st.number_input("Completed weeks to search", min_value=1, max_value=52, value=2, step=1,
                                disabled=scan_scope == "Entire selected backtest")
uploaded_history = st.file_uploader("Optional: replace public membership history with your own dated constituent CSV", type="csv",
                                    help='Columns: date,tickers. One complete roster per date; quote the comma-separated ticker list.')
last_completed = pd.Timestamp(as_of).to_period("W-FRI").end_time.normalize() - pd.Timedelta(weeks=1)
scan_start = (last_completed - pd.Timedelta(weeks=int(lookback_weeks) - 1, days=4) if scan_scope == "Recent completed weeks"
              else window_start.normalize())
scan_end = last_completed
scan_settings = (int(period), average, float(trigger), int(dmi_length), float(adx_cutoff), int(adx_lookback),
                 int(macd_fast), int(macd_slow), int(macd_signal), float(macd_cutoff),
                 str(scan_start), str(scan_end), uploaded_history.getvalue() if uploaded_history else None)
buy_button, sell_button = st.columns(2)
run_buy = buy_button.button("Scan Buy Signals", type="primary")
run_sell = sell_button.button("Scan Sell Signals", type="primary")
if run_buy or run_sell:
    kind = "Buy" if run_buy else "Sell"
    try:
        with st.spinner("Loading historical S&P 500 membership..."):
            history = (parse_membership_history(pd.read_csv(uploaded_history)) if uploaded_history
                       else download_membership_history())
        if scan_start < history["date"].min() or scan_end > history["date"].max():
            raise ValueError(f"Selected scan falls outside membership coverage ({history['date'].min():%Y-%m-%d} to "
                             f"{history['date'].max():%Y-%m-%d}). Shorten the range or supply an updated membership CSV.")
        results, failures, total = scan_sp500(history, scan_start, scan_end, kind, as_of)
        st.session_state[f"{kind}_scan"] = (scan_settings, results, failures, total)
    except Exception as exc:
        st.session_state.pop(f"{kind}_scan", None)
        st.error(f"{kind} scan could not run: {exc}")
for kind in ("Buy", "Sell"):
    stored = st.session_state.get(f"{kind}_scan")
    if stored is None:
        continue
    settings, results, failures, total = stored
    st.subheader(f"S&P 500 {kind} signal results")
    if settings != scan_settings:
        st.info("Settings changed. Run this scan again to update its results.")
        continue
    st.write(f"{len(results)} signals; {total} historically eligible symbols considered; {len(failures)} could not be checked.")
    if results.empty:
        st.info("No qualifying signals found among checked symbols.")
    else:
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.download_button(f"Download {kind} signals (CSV)", results.to_csv(index=False),
                           file_name=f"sp500_{kind.lower()}_signals.csv", mime="text/csv")
    if not failures.empty:
        with st.expander("Symbols with unavailable or insufficient price data"):
            st.dataframe(failures, hide_index=True, use_container_width=True)
            st.download_button(f"Download {kind} data gaps", failures.to_csv(index=False),
                               file_name=f"sp500_{kind.lower()}_data_gaps.csv", mime="text/csv")
st.caption("Volume increased? compares signal-week share volume with the average of the prior three completed weeks; "
           "equality is No, missing data is Unavailable. Full backtest scans can take substantially longer than recent scans. "
           "Public membership history starts in 1996 and is not an audited S&P dataset. Delisted names and reused tickers "
           "can have unavailable or ambiguous Yahoo price histories, so historical coverage is incomplete. "
           "Source: https://github.com/chinobing/historical_sp500_constituents")
