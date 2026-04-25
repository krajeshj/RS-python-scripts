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
BASE_SPRINT = 15
EXTENSION_DAYS = 5
NUM_PICKS = 5
NUM_BACKTESTS = 11

print("Loading data...")
with open(WEB_DATA, "r", encoding="utf-8") as f: web_data = json.load(f)
with open(PRICE_DATA, "r", encoding="utf-8") as f: all_data = json.load(f)

def simulate_dynamic_sprint(picks, capital, entry_idx, use_extension=False):
    risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
    total_pnl = 0
    extend_sprint = False
    
    results = []
    for p in picks:
        t = p["ticker"]
        if t not in all_data: continue
        candles = all_data[t]["candles"]
        if entry_idx >= len(candles): continue
        
        price = candles[entry_idx]["close"]
        lookback = candles[max(0, entry_idx-21):entry_idx]
        low_21 = min([c["low"] for c in lookback]) if lookback else price * 0.97
        stop_p = max(low_21, price * 0.92)
        stop_p = min(stop_p, price * 0.985)
        stop_dist = price - stop_p
        target = price + stop_dist * TARGET_R
        shares = max(1, int(risk_per / stop_dist))
        
        # Initial 15 days
        exit_price, outcome = price, "EXPIRED"
        for d in range(1, BASE_SPRINT + 1):
            if entry_idx + d >= len(candles): break
            c = candles[entry_idx + d]
            if c["low"] <= stop_p: exit_price, outcome = stop_p, "STOP"; break
            if c["high"] >= target: exit_price, outcome = target, "TARGET"; break
            exit_price = c["close"]
            
        # Check for extension if at the end of BASE_SPRINT and still in trade
        if use_extension and outcome == "EXPIRED":
            current_r = (exit_price - price) / stop_dist
            if current_r > 1.5: # "Bagging Profits"
                extend_sprint = True
                # Continue for 5 more days
                for d in range(BASE_SPRINT + 1, BASE_SPRINT + EXTENSION_DAYS + 1):
                    if entry_idx + d >= len(candles): break
                    c = candles[entry_idx + d]
                    if c["low"] <= stop_p: exit_price, outcome = stop_p, "STOP"; break
                    if c["high"] >= target: exit_price, outcome = target, "TARGET"; break
                    exit_price = c["close"]
        
        total_pnl += (exit_price - price) * shares
    return total_pnl

def run_experiment(exp_name, use_extension=False):
    print(f"Running Experiment: {exp_name}...")
    spy_all = all_data["SPY"]["candles"]
    max_idx = len(spy_all) - 22
    test_indices = [max_idx - (i * 20) for i in range(NUM_BACKTESTS)][::-1]
    
    capital = PORTFOLIO_START
    for idx in test_indices:
        # Selection
        s_spy = pd.Series([c["close"] for c in spy_all[:idx]])
        if s_spy.ewm(span=21, adjust=False).mean().iloc[-1] < s_spy.rolling(50).mean().iloc[-1]: continue
        
        candidates = []
        spy_bench = spy_all[idx]["close"] / spy_all[max(0, idx-63)]["close"]
        for t, d in all_data.items():
            if t == "SPY" or d.get("skip_calc", 1): continue
            cnd = d["candles"][:idx]
            if len(cnd) < 63: continue
            c = cnd[-1]["close"]
            if c < 15: continue
            
            # Simplified selection
            rs_score = (c / cnd[-63]["close"]) / spy_bench
            candidates.append({"ticker": t, "rs": rs_score})
        
        candidates.sort(key=lambda x: -x["rs"])
        picks = candidates[:NUM_PICKS]
        if not picks: continue
        capital += simulate_dynamic_sprint(picks, capital, idx, use_extension=use_extension)
        
    total_return = ((capital - PORTFOLIO_START) / PORTFOLIO_START) * 100
    return {"name": exp_name, "return": total_return}

if __name__ == "__main__":
    e1 = run_experiment("CONTROL: Fixed 15-Day Sprint", use_extension=False)
    e2 = run_experiment("EXPT: Dynamic Extension (Bagging > 1.5R)", use_extension=True)
    
    print("\n" + "="*70)
    print(f"{'Experiment':<45} | {'Return %':<10}")
    print("-" * 70)
    for r in [e1, e2]:
        print(f"{r['name']:<45} | {r['return']:>8.1f}%")
    print("="*70)
