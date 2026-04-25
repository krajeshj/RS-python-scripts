import json, os, random, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR) if os.path.basename(DIR) == "scratch" else DIR
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")

# --- Strategy Parameters ---
PORTFOLIO_START = 17000.0
MAX_RISK_PCT = 0.03
TARGET_R = 3.5
SPRINT_DAYS = 15
NUM_PICKS = 5
NUM_BACKTESTS = 11

# Load all data once
with open(WEB_DATA, "r", encoding="utf-8") as f: web_data = json.load(f)
with open(PRICE_DATA, "r", encoding="utf-8") as f: all_data = json.load(f)

# Pre-compute Sector Mapping
sector_etfs = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]
ticker_to_sector_etf = {}
for s in web_data.get("all_stocks", []):
    ticker = s["ticker"]
    sector_name = s["sector"].split(' ')[0] # E.g. "Technology"
    # Simple mapping
    mapping = {
        "Technology": "XLK", "Energy": "XLE", "Financial": "XLF", "Healthcare": "XLV",
        "Consumer": "XLY", "Communication": "XLC", "Basic": "XLB", "Industrials": "XLI",
        "Real": "XLRE", "Utilities": "XLU"
    }
    ticker_to_sector_etf[ticker] = mapping.get(sector_name, "NONE")

def get_rrg_quadrant(tc, sc):
    if len(tc) < 30: return "Unknown"
    rs_raw = tc / sc
    sma_ratio = rs_raw.rolling(14).mean()
    rs_ratio = 100 + ((sma_ratio - sma_ratio.rolling(14).mean()) / sma_ratio.rolling(14).std().replace(0, 1)) * 5
    roc = rs_ratio.diff(10)
    rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
    
    x = rs_ratio.iloc[-1] - 100
    y = rs_mom.iloc[-1] - 100
    
    if x >= 0 and y >= 0: return "Leading"
    if x < 0 and y >= 0: return "Improving"
    if x < 0 and y < 0: return "Lagging"
    return "Weakening"

def get_leading_sectors(idx, spy_all, all_data):
    leading = []
    spy_c = pd.Series([c["close"] for c in spy_all[:idx]])
    for s in sector_etfs:
        if s not in all_data: continue
        tc = pd.Series([c["close"] for c in all_data[s]["candles"][:idx]])
        if len(tc) < 63: continue
        if get_rrg_quadrant(tc, spy_c.tail(len(tc)).reset_index(drop=True)) == "Leading":
            leading.append(s)
    return leading

def simulate_sprint_full(all_data, picks, capital, entry_idx, stop_type="fixed"):
    risk_budget = capital * MAX_RISK_PCT
    risk_per = risk_budget / NUM_PICKS
    trades, total_pnl = [], 0
    for p in picks:
        ticker, price = p["ticker"], p["price"]
        candles = all_data[ticker]["candles"]
        
        if stop_type == "21low":
            lookback = candles[max(0, entry_idx-21):entry_idx]
            lows = [c["low"] for c in lookback]
            low_21 = min(lows) if lows else price * 0.97
            # Ensure stop isn't too tight or too loose
            stop_loss = max(low_21, price * 0.92) # Max 8% stop
            stop_loss = min(stop_loss, price * 0.985) # Min 1.5% stop
            stop_dist = price - stop_loss
        else:
            stop_dist = price * 0.03
            stop_loss = round(price - stop_dist, 2)
            
        target = round(price + stop_dist * TARGET_R, 2)
        shares = max(1, int(risk_per / stop_dist))
        
        exit_price, result = price, "EXPIRED"
        for d in range(1, SPRINT_DAYS + 1):
            if entry_idx + d >= len(candles): break
            l, h, c = candles[entry_idx+d]["low"], candles[entry_idx+d]["high"], candles[entry_idx+d]["close"]
            if l <= stop_loss: exit_price, result = stop_loss, "STOP"; break
            if h >= target: exit_price, result = target, "TARGET"; break
            exit_price = c
        pnl = (exit_price - price) * shares
        total_pnl += pnl
        trades.append({"ticker":ticker, "pnl":pnl, "result":result})
    return trades, total_pnl

def get_wcr(candles):
    # Calculate WCR for a 5-day window ending at the last candle
    if len(candles) < 5: return 0.5
    high = max(c["high"] for c in candles[-5:])
    low = min(c["low"] for c in candles[-5:])
    close = candles[-1]["close"]
    if high == low: return 0.5
    return (close - low) / (high - low)

def run_experiment(exp_name, filter_func, earnings_filter=False, stop_type="fixed"):
    print(f"\n--- Running Experiment: {exp_name} ---")
    spy_all = all_data["SPY"]["candles"]
    max_idx = len(spy_all) - SPRINT_DAYS - 2
    test_indices = [max_idx - (i * SPRINT_DAYS * 2) for i in range(NUM_BACKTESTS)][::-1]
    
    capital, wins, losses = PORTFOLIO_START, 0, 0
    
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        s_spy = pd.Series([c["close"] for c in spy_slice])
        ema21, sma50 = s_spy.ewm(span=21, adjust=False).mean().iloc[-1], s_spy.rolling(50).mean().iloc[-1]
        if ema21 < sma50: continue
        
        leading_sectors = get_leading_sectors(idx, spy_all, all_data)
        mapping = {"Technology": "XLK", "Energy": "XLE", "Financial": "XLF", "Healthcare": "XLV", "Consumer": "XLY", "Communication": "XLC", "Basic": "XLB", "Industrials": "XLI", "Real": "XLRE", "Utilities": "XLU"}
        ticker_to_sector = {s["ticker"]: mapping.get(s["sector"].split(' ')[0], "NONE") for s in web_data.get("all_stocks", [])}

        candidates = []
        spy_bench = s_spy.iloc[-1] / s_spy.iloc[-63] if len(s_spy) > 63 else 1
        for t, d in all_data.items():
            if t in ["SPY"] or d.get("skip_calc",1)==1: continue
            cnd = d.get("candles", [])[:idx]
            if len(cnd) < 63: continue
            tc = pd.Series([c["close"] for c in cnd])
            c = cnd[-1]["close"]
            if c < 15: continue

            # Flow & RMV
            ranges = [o["high"] - o["low"] for o in cnd[-20:]]
            rmv = (sum(ranges)/len(ranges) / c) * 100 * 7.0
            avg_vol = sum([o["volume"] for o in cnd[-20:]]) / 20
            rvol = cnd[-1]["volume"] / avg_vol if avg_vol > 0 else 0
            
            is_leading_sec = ticker_to_sector.get(t, "NONE") in leading_sectors
            quadrant = get_rrg_quadrant(tc, s_spy.tail(len(tc)).reset_index(drop=True))
            
            # WCR Logic
            wcr1 = get_wcr(cnd) # Prior week
            wcr2 = get_wcr(cnd[:-5]) # 2nd week back
            wcr3 = get_wcr(cnd[:-10]) # 3rd week back
            wcr_count = sum(1 for w in [wcr1, wcr2, wcr3] if w > 0.5)

            if not filter_func(quadrant, rvol, rmv, is_leading_sec, wcr1, wcr_count): continue
            
            if earnings_filter:
                future_cnd = all_data[t]["candles"][idx:idx+SPRINT_DAYS]
                has_earnings = False
                for i in range(1, len(future_cnd)):
                    gap = abs(future_cnd[i]["open"] - future_cnd[i-1]["close"]) / future_cnd[i-1]["close"]
                    if gap > 0.05 or (future_cnd[i]["volume"] / avg_vol > 3.5):
                        has_earnings = True; break
                if has_earnings: continue

            rs_score = (c / cnd[-63]["close"]) / spy_bench
            candidates.append({"ticker":t, "price":c, "rs":rs_score})
        
        candidates.sort(key=lambda x: -x["rs"])
        picks = candidates[:NUM_PICKS]
        if not picks: continue

        trades, pnl = simulate_sprint_full(all_data, picks, capital, idx, stop_type=stop_type)
        wins += sum(1 for t in trades if t["pnl"] > 0)
        losses += sum(1 for t in trades if t["pnl"] <= 0)
        capital += pnl
        
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    total_return = ((capital - PORTFOLIO_START) / PORTFOLIO_START * 100)
    print(f"Win Rate: {win_rate:.1f}% | Return: {total_return:.1f}%")
    return {"name": exp_name, "win_rate": win_rate, "return": total_return, "stop": stop_type}

if __name__ == "__main__":
    # Base: Leading Sector + Flow > 1.2 + RMV < 40 + Earnings ON + 21-Day Low Stop
    base_f = lambda q, rvol, rmv, sec, w1, wc: sec and rvol > 1.2 and rmv < 40
    
    print("\n--- WCR EXPERIMENTS (Earnings: ON | Stop: 21low | RMV: 40) ---")
    
    e1 = run_experiment("CONTROL: Base (RMV 40)", 
                        lambda q, rvol, rmv, sec, w1, wc: base_f(q, rvol, rmv, sec, w1, wc), 
                        earnings_filter=True, stop_type="21low")
    
    e2 = run_experiment("TEST A: Prior Week WCR > 50%", 
                        lambda q, rvol, rmv, sec, w1, wc: base_f(q, rvol, rmv, sec, w1, wc) and w1 > 0.5, 
                        earnings_filter=True, stop_type="21low")
    
    e3 = run_experiment("TEST B: 2/3 Weeks WCR > 50%", 
                        lambda q, rvol, rmv, sec, w1, wc: base_f(q, rvol, rmv, sec, w1, wc) and wc >= 2, 
                        earnings_filter=True, stop_type="21low")

    print("\n" + "="*90)
    print(f"{'Experiment':<35} | {'Win Rate':<10} | {'Return':<10}")
    print("-" * 90)
    for e in [e1, e2, e3]:
        print(f"{e['name']:<35} | {e['win_rate']:>8.1f}% | {e['return']:>8.1f}%")
    print("="*90)
