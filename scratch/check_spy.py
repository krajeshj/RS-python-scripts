import yfinance as yf
from datetime import date, timedelta

today = date.today()
yesterday = today - timedelta(days=1)
# Try to get 4/22 explicitly
df = yf.download("SPY", start=date(2026, 4, 15), end=date(2026, 4, 23))
print(df.tail())
