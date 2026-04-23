import json
import pandas as pd
import numpy as np

with open('data/price_history.json', 'r') as f:
    all_data = json.load(f)

# --- Step 1: Calculate RRG quadrants TODAY ---
etfs = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]
etf_names = {
    "XLK": "Technology", "XLE": "Energy", "XLF": "Financials", "XLV": "Healthcare",
    "XLY": "Consumer Disc.", "XLC": "Communication", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLI": "Industrials", "XLRE": "Real Estate", "XLB": "Materials"
}
sector_map = {
    "Technology": "Technology", "Energy": "Energy", "Financials": "Financial Services",
    "Healthcare": "Healthcare", "Consumer Disc.": "Consumer Cyclical",
    "Communication": "Communication Services", "Consumer Staples": "Consumer Defensive",
    "Utilities": "Utilities", "Industrials": "Industrials",
    "Real Estate": "Real Estate", "Materials": "Basic Materials"
}

spy_candles = all_data["SPY"]["candles"]
spy_closes = pd.Series([c["close"] for c in spy_candles])

rrg_today = {}
for pt in etfs:
    t_candles = all_data.get(pt, {}).get("candles", [])
    if len(t_candles) > 30 and len(t_candles) <= len(spy_closes):
        min_len = min(len(t_candles), len(spy_closes))
        tc = pd.Series([c["close"] for c in t_candles[-min_len:]])
        sc = spy_closes.tail(min_len).reset_index(drop=True)
        rs_raw = tc / sc
        sma_ratio = rs_raw.rolling(14).mean()
        rs_ratio = 100 + ((sma_ratio - sma_ratio.rolling(14).mean()) / sma_ratio.rolling(14).std().replace(0, 1)) * 5
        roc = rs_ratio.diff(10)
        rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
        x = round(rs_ratio.iloc[-1] - 100, 2)
        y = round(rs_mom.iloc[-1] - 100, 2)
        # Also get previous point for direction
        x_prev = round(rs_ratio.iloc[-2] - 100, 2)
        y_prev = round(rs_mom.iloc[-2] - 100, 2)
        rrg_today[pt] = {"x": x, "y": y, "dx": x - x_prev, "dy": y - y_prev}

leading = []
improving = []

print("### Sector RRG Quadrants TODAY")
print("| Sector ETF | Sector | x (Ratio) | y (Momentum) | Quadrant | Direction |")
print("|---|---|---|---|---|---|")
for pt in etfs:
    if pt in rrg_today:
        d = rrg_today[pt]
        x, y = d["x"], d["y"]
        dx, dy = d["dx"], d["dy"]
        if x > 0 and y > 0:
            q = "LEADING"
            leading.append(pt)
        elif x < 0 and y > 0:
            q = "IMPROVING"
            improving.append(pt)
        elif x > 0 and y < 0:
            q = "WEAKENING"
        else:
            q = "LAGGING"
        arrow = ""
        if dx > 0 and dy > 0: arrow = "NE (strengthening)"
        elif dx > 0 and dy < 0: arrow = "SE (weakening)"
        elif dx < 0 and dy > 0: arrow = "NW (improving)"
        else: arrow = "SW (deteriorating)"
        print(f"| **{pt}** | {etf_names[pt]} | {x} | {y} | {q} | {arrow} |")

# --- Step 2: Find top RS stocks in Leading + Improving ---
def quarters_rs(closes, closes_ref, n):
    try:
        length = min(len(closes), n * int(252 / 4))
        p_n = closes.tail(length).dropna().head(1).item()
        r_n = closes_ref.tail(length).dropna().head(1).item()
        return (closes.tail(1).item() / p_n) / (closes_ref.tail(1).item() / r_n)
    except:
        return 0

def relative_strength(closes, closes_ref):
    rs1 = quarters_rs(closes, closes_ref, 1)
    rs2 = quarters_rs(closes, closes_ref, 2)
    rs3 = quarters_rs(closes, closes_ref, 3)
    rs4 = quarters_rs(closes, closes_ref, 4)
    return 0.4 * rs1 + 0.2 * rs2 + 0.2 * rs3 + 0.2 * rs4

def calc_atr(candles, n=14):
    if len(candles) < n + 1: return None
    trs = []
    for i in range(-n, 0):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return np.mean(trs)

def calc_sma(candles, n):
    if len(candles) < n: return None
    return np.mean([c['close'] for c in candles[-n:]])

ref_closes = pd.Series([c["close"] for c in spy_candles])

PORTFOLIO = 17000.00
MAX_RISK = PORTFOLIO * 0.03  # $510

def recommend(etf_list, quadrant_name, is_swing):
    target_sectors = set()
    for etf in etf_list:
        if etf_names[etf] in sector_map:
            target_sectors.add(sector_map[etf_names[etf]])
    
    candidates = []
    for ticker, data in all_data.items():
        if ticker == "SPY" or data.get("skip_calc", 1) == 1:
            continue
        if data.get("sector", "") not in target_sectors:
            continue
        candles = data.get("candles", [])
        if len(candles) < 252:
            continue
        
        closes = pd.Series([c["close"] for c in candles])
        rs = relative_strength(closes, ref_closes)
        price = candles[-1]["close"]
        atr = calc_atr(candles)
        sma21 = calc_sma(candles, 21)
        ema8_vals = closes.ewm(span=8, adjust=False).mean()
        ema8 = ema8_vals.iloc[-1]
        sma50 = calc_sma(candles, 50)
        
        if atr is None or sma21 is None or sma50 is None:
            continue
        
        # Extension checks
        ext_from_21 = ((price - sma21) / sma21) * 100
        ext_from_50 = ((price - sma50) / sma50) * 100
        above_ema8 = price > ema8
        above_sma21 = price > sma21
        above_sma50 = price > sma50
        
        # Filter: not too extended, above key MAs
        if ext_from_50 > 15: continue  # too extended from 50 SMA
        if not above_sma50: continue   # must be above 50 SMA
        
        # ATR % for volatility
        atr_pct = (atr / price) * 100
        
        candidates.append({
            "ticker": ticker,
            "name": data.get("name", ticker)[:20],
            "sector": data.get("sector", "")[:15],
            "rs": rs,
            "price": price,
            "atr": atr,
            "atr_pct": atr_pct,
            "sma21": sma21,
            "ema8": ema8,
            "sma50": sma50,
            "ext_21": ext_from_21,
            "ext_50": ext_from_50,
            "above_ema8": above_ema8,
            "above_21": above_sma21,
        })
    
    candidates.sort(key=lambda x: -x["rs"])
    top = candidates[:5]
    
    # Position sizing
    num_picks = len(top)
    if num_picks == 0:
        print(f"\n*No eligible stocks found in {quadrant_name} sectors.*")
        return
    risk_per_pick = MAX_RISK / num_picks
    
    trade_type = "SWING (1-2 wk)" if is_swing else "LONG-TERM"
    print(f"\n### {quadrant_name} - {trade_type} Recommendations (Top 5)")
    sectors_str = ", ".join([f"{etf_names[e]} ({e})" for e in etf_list])
    print(f"Sectors: {sectors_str}")
    print(f"Portfolio: $17,000 | Max Risk: $510 (3%) | Risk/Pick: ${risk_per_pick:.0f}")
    print()
    
    print("| # | Ticker | Name | Price | 8EMA | 21SMA | 50SMA | Ext% | ATR |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(top):
        e8 = ">" if c["above_ema8"] else "<"
        s21 = ">" if c["above_21"] else "<"
        print(f"| {i+1} | **{c['ticker']}** | {c['name']} | ${c['price']:.2f} | {e8}${c['ema8']:.2f} | {s21}${c['sma21']:.2f} | ${c['sma50']:.2f} | {c['ext_50']:+.1f}% | ${c['atr']:.2f} |")
    
    print()
    print("#### OCO Order Details (Power E*TRADE: Exit Plan)")
    if is_swing:
        print("*Stop: 1 ATR or max 6%. Target: 3R.*")
    else:
        print("*Stop: 1.5 ATR. Target: 3R.*")
    print()
    print("| # | Ticker | Action | Shares | Entry (Limit) | Stop Loss (Sell-Stop) | Profit Target (Sell-Limit) | Risk $ | R:R |")
    print("|---|---|---|---|---|---|---|---|---|")
    
    total_cost = 0
    for i, c in enumerate(top):
        price = c["price"]
        atr = c["atr"]
        
        if is_swing:
            stop_dist = min(atr, price * 0.06)
            r_mult = 3
        else:
            stop_dist = 1.5 * atr
            r_mult = 3
        
        stop_loss = round(price - stop_dist, 2)
        target = round(price + (stop_dist * r_mult), 2)
        sl_pct = round((stop_dist / price) * 100, 1)
        
        shares = int(risk_per_pick / stop_dist)
        if shares == 0: shares = 1
        cost = shares * price
        if total_cost + cost > PORTFOLIO:
            shares = max(1, int((PORTFOLIO - total_cost) / price))
            cost = shares * price
        total_cost += cost
        
        actual_risk = shares * stop_dist
        
        print(f"| {i+1} | **{c['ticker']}** | BUY | {shares} | ${price:.2f} | ${stop_loss:.2f} (-{sl_pct}%) | ${target:.2f} (+{round((target-price)/price*100,1)}%) | ${actual_risk:.0f} | 1:{r_mult} |")
    
    print(f"\n**Total Capital Deployed:** ${total_cost:,.2f} of $17,000")

print()
if leading:
    recommend(leading, "LEADING", is_swing=True)
    recommend(leading, "LEADING", is_swing=False)

if improving:
    recommend(improving, "IMPROVING", is_swing=True)
    recommend(improving, "IMPROVING", is_swing=False)
