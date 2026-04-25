import json, os
import pandas as pd

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR)
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

def verify():
    with open(WEB_DATA, "r") as f: web_data = json.load(f)
    with open(PRICE_DATA, "r") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    spy_c = pd.Series([c["close"] for c in spy_all])
    
    mapping = {"Technology": "XLK", "Energy": "XLE", "Financial": "XLF", "Healthcare": "XLV", "Consumer": "XLY", "Communication": "XLC", "Basic": "XLB", "Industrials": "XLI", "Real": "XLRE", "Utilities": "XLU"}
    
    # Identify Leading Sectors
    leading_sectors = []
    for s_etf in mapping.values():
        if s_etf not in all_data: continue
        tc = pd.Series([c["close"] for c in all_data[s_etf]["candles"]])
        if get_rrg_quadrant(tc, spy_c.tail(len(tc)).reset_index(drop=True)) == "Leading":
            leading_sectors.append(s_etf)
    
    print(f"Leading Sectors: {leading_sectors}")

    tickers = ["BE", "ERA", "AXTI", "SNDK", "ANL"]
    print(f"\n{'Ticker':<6} | {'Price':<6} | {'RMV':<6} | {'RVol':<6} | {'DTE':<4} | {'Sector':<10} | {'Leading?':<8} | {'RS':<6}")
    print("-" * 75)
    
    for t in tickers:
        s = next((x for x in web_data["all_stocks"] if x["ticker"] == t), None)
        if not s or t not in all_data: continue
        
        cnd = all_data[t]["candles"]
        c = cnd[-1]["close"]
        ranges = [o["high"] - o["low"] for o in cnd[-20:]]
        rmv = (sum(ranges)/len(ranges) / c) * 100 * 7.0
        avg_vol = sum([o["volume"] for o in cnd[-20:]]) / 20
        rvol = cnd[-1]["volume"] / avg_vol
        
        sector_etf = mapping.get(s["sector"].split(' ')[0], "NONE")
        is_leading = sector_etf in leading_sectors
        dte = s.get("days_to_earnings", -1)
        
        print(f"{t:<6} | {c:<6.2f} | {rmv:<6.1f} | {rvol:<6.1f} | {dte:<4} | {sector_etf:<10} | {str(is_leading):<8} | {s.get('rs',0):<6.1f}")

if __name__ == "__main__":
    verify()
