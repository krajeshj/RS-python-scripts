import json
import pandas as pd
import numpy as np

with open('data/price_history.json', 'r') as f:
    all_data = json.load(f)

# =============================================================================
# MATH: Triple $17,000 → $51,000 in 1 year (252 trading days)
# =============================================================================
# ~25 swing cycles/year (10 trading days each)
# Risk 3% per cycle, need ~4.2% expected gain per cycle to compound to 3x
# Required: 5R target with ~40% win rate
# EV per cycle = 0.40 * 5R - 0.60 * 1R = 2.0R - 0.6R = 1.4R
# 1.4R * 3% risk = 4.2% gain per cycle
# $17,000 * 1.042^25 = $47,700 (~2.8x)  
# Push to 5R with 45% win rate: 
# 0.45 * 5 - 0.55 = 1.7R → 5.1% per cycle → $17,000 * 1.051^25 = $58,800 (3.5x)
# =============================================================================

PORTFOLIO = 17000.00
MAX_RISK_PCT = 0.03
TARGET_R = 5  # 5R target (was 3R)
CYCLES_PER_YEAR = 25
REQUIRED_WIN_RATE = 0.40

print("=" * 70)
print("AGGRESSIVE SCREENER: Triple Principal Strategy")
print("=" * 70)
print(f"Starting Capital: ${PORTFOLIO:,.0f}")
print(f"Annual Target:    ${PORTFOLIO * 3:,.0f} (3x)")
print(f"Risk Per Cycle:   {MAX_RISK_PCT*100:.0f}% (${PORTFOLIO * MAX_RISK_PCT:,.0f})")
print(f"R:R Ratio:        1:{TARGET_R}")
print(f"Cycles/Year:      ~{CYCLES_PER_YEAR} (every 10 trading days)")
ev = REQUIRED_WIN_RATE * TARGET_R - (1 - REQUIRED_WIN_RATE)
ev_pct = ev * MAX_RISK_PCT * 100
compound = PORTFOLIO * (1 + ev * MAX_RISK_PCT) ** CYCLES_PER_YEAR
print(f"Required Win Rate: >{REQUIRED_WIN_RATE*100:.0f}%")
print(f"Expected Value:   {ev:.1f}R per cycle ({ev_pct:.1f}% portfolio)")
print(f"Projected EOY:    ${compound:,.0f} ({compound/PORTFOLIO:.1f}x)")
print("=" * 70)

# --- RRG Quadrants ---
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

leading = []
improving = []
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
        x = rs_ratio.iloc[-1] - 100
        y = rs_mom.iloc[-1] - 100
        if x > 0 and y > 0: leading.append(pt)
        elif x < 0 and y > 0: improving.append(pt)

# --- RS Calculation ---
def quarters_rs(closes, closes_ref, n):
    try:
        length = min(len(closes), n * 63)
        p_n = closes.tail(length).dropna().head(1).item()
        r_n = closes_ref.tail(length).dropna().head(1).item()
        return (closes.iloc[-1] / p_n) / (closes_ref.iloc[-1] / r_n)
    except:
        return 0

def relative_strength(closes, closes_ref):
    return 0.4 * quarters_rs(closes, closes_ref, 1) + 0.2 * quarters_rs(closes, closes_ref, 2) + 0.2 * quarters_rs(closes, closes_ref, 3) + 0.2 * quarters_rs(closes, closes_ref, 4)

def calc_atr(candles, n=14):
    if len(candles) < n + 1: return None
    trs = []
    for i in range(-n, 0):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return np.mean(trs)

# --- Aggressive screening ---
def screen_aggressive(etf_list, quadrant_name):
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
        rs = relative_strength(closes, spy_closes)
        price = candles[-1]["close"]
        atr = calc_atr(candles)
        if atr is None: continue
        
        sma50 = np.mean([c["close"] for c in candles[-50:]])
        sma21 = np.mean([c["close"] for c in candles[-21:]])
        ema8 = closes.ewm(span=8, adjust=False).mean().iloc[-1]
        sma200 = np.mean([c["close"] for c in candles[-200:]])
        
        # 52-week high proximity
        high_52w = max(c["high"] for c in candles[-252:])
        pct_from_high = ((price - high_52w) / high_52w) * 100
        
        ext_50 = ((price - sma50) / sma50) * 100
        atr_pct = (atr / price) * 100
        
        # AGGRESSIVE FILTERS for 5R potential:
        # 1. Must be above ALL key MAs (strong uptrend)
        if price < sma50 or price < sma200: continue
        # 2. Not overextended (need room to run 5R)
        if ext_50 > 12: continue
        # 3. Within 15% of 52-week high (leaders, not laggards)
        if pct_from_high < -15: continue
        # 4. Must have momentum (price > 8 EMA)
        if price < ema8: continue
        # 5. Tight volatility for clean entries
        if atr_pct > 5: continue
        
        # Score: weight RS heavily + proximity to high + low volatility
        momentum_score = rs * 100 + (100 + pct_from_high) - atr_pct * 5
        
        candidates.append({
            "ticker": ticker,
            "name": data.get("name", ticker)[:20],
            "sector": data.get("sector", "")[:15],
            "rs": rs,
            "price": price,
            "atr": atr,
            "atr_pct": atr_pct,
            "ema8": ema8,
            "sma21": sma21,
            "sma50": sma50,
            "ext_50": ext_50,
            "high_52w": high_52w,
            "pct_high": pct_from_high,
            "score": momentum_score
        })
    
    candidates.sort(key=lambda x: -x["score"])
    top = candidates[:5]
    
    if not top:
        print(f"\n*No stocks passed aggressive filters in {quadrant_name}.*")
        return
    
    risk_per_pick = (PORTFOLIO * MAX_RISK_PCT) / len(top)
    
    print(f"\n### {quadrant_name} Sectors - 5R Swing Trades")
    sectors_str = ", ".join([f"{etf_names[e]}" for e in etf_list])
    print(f"Sectors: {sectors_str}")
    print()
    
    print("| # | Ticker | Price | vs 52wH | 8EMA | 21SMA | 50SMA | Ext% | ATR% |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(top):
        print(f"| {i+1} | **{c['ticker']}** | ${c['price']:.2f} | {c['pct_high']:+.1f}% | ${c['ema8']:.2f} | ${c['sma21']:.2f} | ${c['sma50']:.2f} | {c['ext_50']:+.1f}% | {c['atr_pct']:.1f}% |")
    
    print()
    print("#### OCO Orders (5R Target)")
    print("| # | Ticker | BUY | Shares | Stop Loss | Profit Target (5R) | Risk $ | Potential $ |")
    print("|---|---|---|---|---|---|---|---|")
    
    total_cost = 0
    total_potential = 0
    for i, c in enumerate(top):
        price = c["price"]
        atr = c["atr"]
        
        # Tight stop: 1 ATR or max 4% (tighter for 5R)
        stop_dist = min(atr, price * 0.04)
        stop_loss = round(price - stop_dist, 2)
        target = round(price + (stop_dist * TARGET_R), 2)
        sl_pct = round((stop_dist / price) * 100, 1)
        tgt_pct = round((stop_dist * TARGET_R / price) * 100, 1)
        
        shares = int(risk_per_pick / stop_dist)
        if shares == 0: shares = 1
        cost = shares * price
        if total_cost + cost > PORTFOLIO:
            shares = max(1, int((PORTFOLIO - total_cost) / price))
            cost = shares * price
        total_cost += cost
        
        potential = shares * stop_dist * TARGET_R
        total_potential += potential
        actual_risk = shares * stop_dist
        
        print(f"| {i+1} | **{c['ticker']}** | ${price:.2f} | {shares} | ${stop_loss:.2f} (-{sl_pct}%) | ${target:.2f} (+{tgt_pct}%) | ${actual_risk:.0f} | +${potential:.0f} |")
    
    print(f"\n**Capital:** ${total_cost:,.0f} | **Total Risk:** ${PORTFOLIO * MAX_RISK_PCT:,.0f} (3%) | **Max Potential:** +${total_potential:,.0f} (+{total_potential/PORTFOLIO*100:.1f}%)")

print()
if leading:
    screen_aggressive(leading, "LEADING")
if improving:
    screen_aggressive(improving, "IMPROVING")

# Projection table
print("\n### Compounding Projection (25 cycles, 40% win rate @ 5R)")
print("| Cycle | Capital | Risk (3%) | Win P&L | Loss P&L | Expected |")
print("|---|---|---|---|---|---|")
capital = PORTFOLIO
for cycle in range(1, 26):
    risk = capital * MAX_RISK_PCT
    win_pnl = risk * TARGET_R
    loss_pnl = -risk
    expected = REQUIRED_WIN_RATE * win_pnl + (1 - REQUIRED_WIN_RATE) * loss_pnl
    if cycle <= 5 or cycle % 5 == 0 or cycle == 25:
        print(f"| {cycle} | ${capital:,.0f} | ${risk:,.0f} | +${win_pnl:,.0f} | -${abs(loss_pnl):,.0f} | +${expected:,.0f} |")
    capital += expected
print(f"\n**Final projected capital after 25 cycles: ${capital:,.0f} ({capital/PORTFOLIO:.1f}x)**")
