import json
d = json.load(open('output/sprint_data.json'))
trades = [t for b in d['backtests'] for t in b.get('trades',[])]
results = {}
for t in trades:
    r = t['result']
    results[r] = results.get(r, 0) + 1
print("Result breakdown:", results)

for r in results:
    subset = [t for t in trades if t['result'] == r]
    avg_pnl = sum(t['pnl'] for t in subset) / len(subset)
    avg_pct = sum(t['pnl_pct'] for t in subset) / len(subset)
    print(f"  {r}: count={len(subset)}, avg P&L=${avg_pnl:.2f}, avg %={avg_pct:.1f}%")
