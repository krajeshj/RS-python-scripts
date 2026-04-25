import json, os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

print("Loading data...")
with open(WEB_DATA, "r", encoding="utf-8") as f: web_data = json.load(f)
with open(PRICE_DATA, "r", encoding="utf-8") as f: all_data = json.load(f)

def simulate_sprint(picks, capital, entry_idx, trailing_stop=False):
    risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
    total_pnl = 0
    
    for p in picks:
        t = p["ticker"]
        if t not in all_data: continue
        candles = all_data[t]["candles"]
        if entry_idx >= len(candles): continue
        
        price = candles[entry_idx]["close"]
        
        # Initial Stop (21-day low)
        lookback = candles[max(0, entry_idx-21):entry_idx]
        lows = [c["low"] for c in lookback]
        initial_low_21 = min(lows) if lows else price * 0.97
        stop_loss = max(initial_low_21, price * 0.92)
        stop_loss = min(stop_loss, price * 0.985)
        
        stop_dist = price - stop_loss
        target = price + stop_dist * TARGET_R
        shares = max(1, int(risk_per / stop_dist))
        
        exit_price = price
        for d in range(1, SPRINT_DAYS + 1):
            if entry_idx + d >= len(candles): break
            candle = candles[entry_idx + d]
            
            # Trailing Stop Logic: Update stop to new 21-day low if it has moved up
            if trailing_stop:
                current_lookback = candles[max(0, entry_idx+d-21):entry_idx+d]
                current_lows = [c["low"] for c in current_lookback]
                new_low_21 = min(current_lows) if current_lows else stop_loss
                # Only move stop UP, never down
                stop_loss = max(stop_loss, new_low_21)

            if candle["low"] <= stop_loss: 
                exit_price = stop_loss
                break
            if candle["high"] >= target: 
                exit_price = target
                break
            exit_price = candle["close"]
        
        total_pnl += (exit_price - price) * shares
    return total_pnl

def run_experiment(exp_name, trailing_stop=False):
    print(f"Running Experiment: {exp_name}...")
    spy_all = all_data["SPY"]["candles"]
    max_idx = len(spy_all) - SPRINT_DAYS - 1
    test_indices = [max_idx - (i * SPRINT_DAYS) for i in range(NUM_BACKTESTS)][::-1]
    
    capital = PORTFOLIO_START
    
    for idx in test_indices:
        # Selection Logic (Simplified Top RS from Leading Sectors)
        s_spy = pd.Series([c["close"] for c in spy_all[:idx]])
        if s_spy.ewm(span=21, adjust=False).mean().iloc[-1] < s_spy.rolling(50).mean().iloc[-1]: continue
        
        candidates = []
        spy_bench = spy_all[idx]["close"] / spy_all[max(0, idx-63)]["close"]
        for t, d in all_data.items():
            if t == "SPY" or d.get("skip_calc", 1): continue
            cnd = d["candles"][:idx]
            if len(cnd) < 63: continue
            
            # Flow / RMV filter
            c = cnd[-1]["close"]
            ranges = [o["high"] - o["low"] for o in cnd[-20:]]
            rmv = (sum(ranges)/len(ranges) / c) * 100 * 7.0
            avg_vol = sum([o["volume"] for o in cnd[-20:]]) / 20
            rvol = cnd[-1]["volume"] / avg_vol if avg_vol > 0 else 0
            
            if rvol > 1.2 and rmv < 40:
                rs_score = (c / cnd[-63]["close"]) / spy_bench
                candidates.append({"ticker": t, "rs": rs_score})
        
        candidates.sort(key=lambda x: -x["rs"])
        picks = candidates[:NUM_PICKS]
            
        if not picks: continue
        capital += simulate_sprint(picks, capital, idx, trailing_stop=trailing_stop)
        
    total_return = ((capital - PORTFOLIO_START) / PORTFOLIO_START) * 100
    return {"name": exp_name, "return": total_return}

if __name__ == "__main__":
    e1 = run_experiment("CONTROL: Fixed 21-Day Low Stop", trailing_stop=False)
    e2 = run_experiment("EXPT: Trailing 21-Day Low Stop", trailing_stop=True)
    
    print("\n" + "="*70)
    print(f"{'Experiment':<45} | {'Return %':<10}")
    print("-" * 70)
    for r in [e1, e2]:
        print(f"{r['name']:<45} | {r['return']:>8.1f}%")
    print("="*70)
