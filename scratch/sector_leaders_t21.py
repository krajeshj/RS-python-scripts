import json
import pandas as pd
import numpy as np

DIR = '.'
LOOKBACK = 21

with open('data/price_history.json', 'r') as f:
    all_data = json.load(f)

# --- Step 1: Calculate RRG quadrants at T-21 for sector ETFs ---
etfs = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]
etf_names = {
    "XLK": "Technology", "XLE": "Energy", "XLF": "Financials", "XLV": "Healthcare",
    "XLY": "Consumer Disc.", "XLC": "Communication", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLI": "Industrials", "XLRE": "Real Estate", "XLB": "Materials"
}

spy_candles = all_data.get("SPY", {}).get("candles", [])
# Trim to T-21 (remove last 21 candles to simulate "standing at T-21")
spy_candles_t21 = spy_candles[:-LOOKBACK]
spy_closes = pd.Series([c["close"] for c in spy_candles_t21])

rrg_at_t21 = {}
for pt in etfs:
    t_candles = all_data.get(pt, {}).get("candles", [])
    t_candles_t21 = t_candles[:-LOOKBACK]
    if len(t_candles_t21) > 30 and len(t_candles_t21) <= len(spy_closes):
        try:
            min_len = min(len(t_candles_t21), len(spy_closes))
            tc = pd.Series([c["close"] for c in t_candles_t21[-min_len:]])
            sc = spy_closes.tail(min_len).reset_index(drop=True)
            rs_raw = tc / sc
            sma_ratio = rs_raw.rolling(14).mean()
            rs_ratio = 100 + ((sma_ratio - sma_ratio.rolling(14).mean()) / sma_ratio.rolling(14).std().replace(0, 1)) * 5
            roc = rs_ratio.diff(10)
            rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
            x = round(rs_ratio.iloc[-1] - 100, 2)
            y = round(rs_mom.iloc[-1] - 100, 2)
            rrg_at_t21[pt] = {"x": x, "y": y}
        except:
            pass

# Classify quadrants
leading = []   # x > 0, y > 0
improving = [] # x < 0, y > 0
weakening = [] # x > 0, y < 0
lagging = []   # x < 0, y < 0

print("### Sector RRG Quadrants at T-21 (21 Trading Days Ago)")
print("| Sector ETF | Sector | RS-Ratio (x) | RS-Momentum (y) | Quadrant |")
print("|---|---|---|---|---|")
for pt in etfs:
    if pt in rrg_at_t21:
        x = rrg_at_t21[pt]["x"]
        y = rrg_at_t21[pt]["y"]
        if x > 0 and y > 0:
            q = "LEADING"
            leading.append(pt)
        elif x < 0 and y > 0:
            q = "IMPROVING"
            improving.append(pt)
        elif x > 0 and y < 0:
            q = "WEAKENING"
            weakening.append(pt)
        else:
            q = "LAGGING"
            lagging.append(pt)
        print(f"| **{pt}** | {etf_names[pt]} | {x} | {y} | {q} |")

# --- Step 2: Map sectors to stocks, calculate RS at T-21 ---
# Map ETF sector name to sector strings in price_history
sector_map = {
    "Technology": "Technology", "Energy": "Energy", "Financials": "Financial Services",
    "Healthcare": "Healthcare", "Consumer Disc.": "Consumer Cyclical",
    "Communication": "Communication Services", "Consumer Staples": "Consumer Defensive",
    "Utilities": "Utilities", "Industrials": "Industrials",
    "Real Estate": "Real Estate", "Materials": "Basic Materials"
}

def quarters_rs(closes, closes_ref, n):
    try:
        length = min(len(closes), n * int(252 / 4))
        df_p = closes.tail(length).dropna()
        p_n = df_p.head(1).item()
        df_r = closes_ref.tail(length).dropna()
        r_n = df_r.head(1).item()
        p = closes.tail(1).item()
        r = closes_ref.tail(1).item()
        return (p / p_n) / (r / r_n)
    except:
        return 0

def relative_strength(closes, closes_ref):
    rs1 = quarters_rs(closes, closes_ref, 1)
    rs2 = quarters_rs(closes, closes_ref, 2)
    rs3 = quarters_rs(closes, closes_ref, 3)
    rs4 = quarters_rs(closes, closes_ref, 4)
    return 0.4 * rs1 + 0.2 * rs2 + 0.2 * rs3 + 0.2 * rs4

ref_closes_t21 = pd.Series([c["close"] for c in spy_candles_t21])

def find_top_stocks(etf_list, quadrant_name):
    target_sectors = set()
    for etf in etf_list:
        name = etf_names[etf]
        if name in sector_map:
            target_sectors.add(sector_map[name])
    
    candidates = []
    for ticker, data in all_data.items():
        if ticker == "SPY" or data.get("skip_calc", 1) == 1:
            continue
        sector = data.get("sector", "")
        if sector not in target_sectors:
            continue
        
        candles = data.get("candles", [])
        candles_t21 = candles[:-LOOKBACK]
        if len(candles_t21) < 252:
            continue
        
        closes_t21 = pd.Series([c["close"] for c in candles_t21])
        rs = relative_strength(closes_t21, ref_closes_t21)
        
        # Current price (today) and entry price (T-21)
        entry_price = candles_t21[-1]["close"]
        current_price = candles[-1]["close"]
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        candidates.append({
            "ticker": ticker,
            "name": data.get("name", ticker)[:20],
            "sector": sector[:15],
            "rs": round(rs, 4),
            "entry": round(entry_price, 2),
            "current": round(current_price, 2),
            "pnl_pct": round(pnl_pct, 1)
        })
    
    # Rank by RS and take top 10
    candidates.sort(key=lambda x: -x["rs"])
    
    # Assign percentile rank
    total = len(candidates)
    for i, c in enumerate(candidates):
        c["rs_pct"] = round((1 - i / total) * 100, 1) if total > 0 else 0
    
    top = candidates[:10]
    
    print(f"\n### Top 10 Stocks in {quadrant_name} Sectors at T-21 (and 21-day forward P&L)")
    sectors_str = ", ".join([f"{etf_names[e]} ({e})" for e in etf_list])
    print(f"Sectors: {sectors_str}")
    print(f"Universe: {total} stocks in these sectors")
    print()
    print("| Rank | Ticker | Name | Sector | RS Score | Entry (T-21) | Today | 21d P&L |")
    print("|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(top):
        sign = "+" if c["pnl_pct"] >= 0 else ""
        print(f"| {i+1} | **{c['ticker']}** | {c['name']} | {c['sector']} | {c['rs']:.4f} | ${c['entry']} | ${c['current']} | {sign}{c['pnl_pct']}% |")
    
    # Summary stats
    if top:
        avg_pnl = np.mean([c["pnl_pct"] for c in top])
        winners = sum(1 for c in top if c["pnl_pct"] > 0)
        print(f"\n**Avg 21d Return:** {avg_pnl:+.1f}% | **Win Rate:** {winners}/{len(top)}")
    
    return top

print()
if leading:
    lt_stocks = find_top_stocks(leading, "LEADING")
else:
    print("\n*No sectors were in the LEADING quadrant at T-21.*")

if improving:
    imp_stocks = find_top_stocks(improving, "IMPROVING")
else:
    print("\n*No sectors were in the IMPROVING quadrant at T-21.*")
