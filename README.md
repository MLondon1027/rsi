# Weekly Buy and Sell dashboard

Run with Python 3.10+:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The ticker and backtest years control the single-stock Buy and Sell sections. Data are split/dividend adjusted daily OHLC and share volume from Yahoo Finance, aggregated into completed Friday-ending weeks. Dates are the week's last available trading session. The most recent calendar week is excluded until the following Monday.

## Buy signals

Weekly RSI crosses from at or below the chosen level to above it. Default: Wilder's RSI(14), crossing above 36.0, reference 70.0. Wilder's, Simple, Exponential, Weighted, and Hull averages are selectable. The Buy table adds weekly volume, the average volume of the preceding three completed weeks, and Volume increased? Yes means strictly greater than that average. No means less than or equal. Missing volume or insufficient history is Unavailable. Volume does not filter out Buy signals.

## Sell signals

Weekly DI- crosses above DI+ with either same-week ADX below the selected cutoff (default 13.5), or ADX above its value N weeks earlier (default 4). Simple rolling directional movement/true-range sums produce DI; ADX uses a simple DX average with the same period (default 14).

MACD uses adjustable exponential fast 12 / slow 26 / signal 9. MACD minus signal is included only at existing DMI sell dates. True sell requires both an existing DMI sell and a same-week MACD downward cross, plus MACD minus signal strictly below the cutoff (default **-1.0**). Equality does not qualify. A red outline marks confirmed True sells; the original DMI candidates remain listed.

## S&P 500 Buy and Sell scanner

Use Scan Buy Signals or Scan Sell Signals. Recent completed weeks defaults to 2. Entire selected backtest scans the sidebar's selected number of years. Both scanners use dated constituent snapshots and admit a signal only if the ticker was an index member on that signal date. The union of past members is searched, including removed constituents. The Sell scanner returns confirmed True sells; the Buy scanner also includes its volume check.

Public history is fetched from https://github.com/chinobing/historical_sp500_constituents and starts on 1996-01-02. Scans outside the dataset's available dates stop with a coverage message. You can supply your own CSV with date,tickers columns and complete dated rosters; quote each comma-separated ticker list. A roster applies from its date until the next dated roster.

The public history is not an audited S&P feed. Delisted symbols, ticker reuse, and incomplete Yahoo price history prevent a fully survivorship-free historical backtest. Unavailable or insufficient ticker data are reported and downloadable. The app does not silently replace past members with current stocks. Partial gaps and ticker identity errors may remain; validate with an authoritative constituent and security-identity dataset before relying on historical results. Full multi-year scans can take substantially longer than recent scans.
