import json
from datetime import datetime, timezone

with open('data/price_history.json', 'r') as f:
    d = json.load(f)

print(f"Total Tickers: {len(d)}")
if 'SPY' in d:
    last_candle = d['SPY']['candles'][-1]
    last_date = datetime.fromtimestamp(last_candle['datetime'], tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"Latest Date in SPY: {last_date}")
else:
    print("SPY not found in data")

# List some tickers to see if it's just NQ100
tickers = list(d.keys())
print(f"Sample tickers: {tickers[:20]}")
