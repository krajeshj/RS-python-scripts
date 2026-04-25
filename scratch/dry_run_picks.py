import json, os
import pandas as pd

ROOT = os.getcwd()
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")

def get_rrg_quadrant(tc, sc):
    if len(tc) < 30: return "Unknown"
    rs_raw = tc / sc
    sma_ratio = rs_raw.rolling(14).mean()
    rs_ratio = 100 + ((sma_ratio - sma_ratio.rolling(14).mean()) / sma_ratio.rolling(14).std().replace(0, 1)) * 5
    roc = rs_ratio.diff(10)
    rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
    x, y = rs_ratio.iloc[-1] - 100, rs_mom.iloc[-1] - 100
    if x >= 0 and y >= 0: return "Leading"
    return "Other"

def dry_run():
    with open(WEB_DATA, "r") as f: web_data = json.load(f)
    with open(PRICE_DATA, "r") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    spy_c = pd.Series([c["close"] for c in spy_all])
    
    mapping = {"Technology": "XLK", "Energy": "XLE", "Financial": "XLF", "Healthcare": "XLV", "Consumer": "XLY", "Communication": "XLC", "Basic": "XLB", "Industrials": "XLI", "Real": "XLRE", "Utilities": "XLU"}
    leading_sectors = [s for s in mapping.values() if s in all_data and get_rrg_quadrant(pd.Series([c["close"] for c in all_data[s]["candles"]]), spy_c.tail(len(all_data[s]["candles"])).reset_index(drop=True)) == "Leading"]
    
    final_candidates = []
    for s in web_data["all_stocks"]:
        t = s["ticker"]
        if t not in all_data: continue
        cnd = all_data[t]["candles"]
        if len(cnd) < 63: continue
        c = cnd[-1]["close"]
        if c < 15: continue
        
        ranges = [o["high"] - o["low"] for o in cnd[-20:]]
        rmv = (sum(ranges)/len(ranges) / c) * 100 * 7.0
        avg_vol = sum([o["volume"] for o in cnd[-20:]]) / 20
        rvol = cnd[-1]["volume"] / avg_vol if avg_vol > 0 else 0
        
        sector_etf = mapping.get(s["sector"].split(' ')[0], "NONE")
        is_leading = sector_etf in leading_sectors
        dte = s.get("days_to_earnings", -1)
        earnings_safe = (dte >= 15 or dte == -1)

        if is_leading and rvol > 1.2 and rmv < 40 and earnings_safe:
            s["dry_rmv"] = rmv
            s["dry_rvol"] = rvol
            final_candidates.append(s)
            
    final_candidates.sort(key=lambda x: -x.get("rs", 0))
    
    print(f"Top {len(final_candidates[:5])} REAL Picks under Champion Filter:")
    for p in final_candidates[:5]:
        print(f"{p['ticker']:<6} | RS: {p.get('rs',0):<4} | RMV: {p['dry_rmv']:<5.1f} | RVol: {p['dry_rvol']:<4.1f} | Sector: {p['sector']}")

if __name__ == "__main__":
    dry_run()
