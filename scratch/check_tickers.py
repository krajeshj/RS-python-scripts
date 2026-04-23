import json
import pandas as pd

with open('output/web_data.json', 'r') as f:
    web_data = json.load(f)

stocks = web_data.get('stocks', []) + web_data.get('all_stocks', []) + web_data.get('pulse', []) + web_data.get('tips', [])
seen = set()
unique = []
for s in stocks:
    if s['ticker'] not in seen:
        seen.add(s['ticker'])
        unique.append(s)

check = ['BW', 'VALE', 'TIGO']
print('| Ticker | RS Rank | 1W RS | 1M RS | RMV | CANSLIM | Highlights |')
print('|---|---|---|---|---|---|---|')
for s in unique:
    if s['ticker'] in check:
        c = s.get('canslim', {})
        score = sum([c.get('c',False), c.get('a',False), c.get('n',False), c.get('s',False), c.get('i',False), c.get('l',False), c.get('m',False)])
        print(f"| {s['ticker']} | {s.get('rs')} | {s.get('rs_1w_pct')} | {s.get('rs_1m_pct')} | {s.get('rmv')} | {score} | {s.get('highlights')} |")
