import json

with open('output/web_data.json', 'r') as f:
    d = json.load(f)

stocks = d.get("stocks", [])
print(f"Total Stocks in web_data: {len(stocks)}")
if stocks:
    print(f"First 10 tickers: {[s.get('ticker') for s in stocks[:10]]}")
