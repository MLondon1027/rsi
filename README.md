# Weekly RSI crossings dashboard

Run with Python 3.10+:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a ticker and a backtest length in years. The chart and CSV list weeks when the 14-week Wilder RSI moves from at or below 36 to above 36. The 70 level is shown as an overbought reference.

The app downloads daily adjusted closes from Yahoo Finance through `yfinance` and groups them into Friday-ending weeks. A week is included only after the following Monday begins, so live prices cannot generate a provisional signal. Each displayed date is the last available trading session in that week. The app fetches two additional years for RSI warmup. Network access is required when running it.
