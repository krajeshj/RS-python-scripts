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

sector_mapping = {
    "Technology": "XLK", "Energy": "XLE", "Financial": "XLF", "Healthcare": "XLV",
    "Consumer": "XLY", "Communication": "XLC", "Basic": "XLB", "Industrials": "XLI",
    "Real": "XLRE", "Utilities": "XLU"
}

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
    for s in ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]:
        if s not in all_data: continue
        tc = pd.Series([c["close"] for c in all_data[s]["candles"][:idx]])
        if len(tc) < 63: continue
        if get_rrg_quadrant(tc, spy_c.tail(len(tc)).reset_index(drop=True)) == "Leading":
            leading.append(s)
    return leading

def simulate_sprint(picks, capital, entry_idx, use_sector_exit=False):
    risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
    total_pnl = 0
    spy_candles = all_data["SPY"]["candles"]
    
    for p in picks:
        t = p["ticker"]
        sector_etf = p["sector_etf"]
        candles = all_data[t]["candles"]
        
        price = candles[entry_idx]["close"]
        stop_dist = price * 0.03
        stop_loss = price - stop_dist
        target = price + stop_dist * TARGET_R
        shares = max(1, int(risk_per / stop_dist))
        
        exit_price = price
        for d in range(1, SPRINT_DAYS + 1):
            if entry_idx + d >= len(candles): break
            
            # Sector Exit Check
            if use_sector_exit and sector_etf in all_data:
                sec_cnd = all_data[sector_etf]["candles"][:entry_idx+d+1]
                spy_cnd = spy_candles[:entry_idx+d+1]
                
                s_sec = pd.Series([c["close"] for c in sec_cnd])
                s_spy = pd.Series([c["close"] for c in spy_cnd])
                
                # Relative Strength Line (Sector / SPY)
                rs_line = s_sec / s_spy
                ema9 = rs_line.ewm(span=9, adjust=False).mean().iloc[-1]
                sma21 = rs_line.rolling(21).mean().iloc[-1]
                
                if ema9 < sma21:
                    exit_price = candles[entry_idx+d]["close"]
                    # print(f"  [{t}] Sector Exit Triggered at day {d}")
                    break
            
            candle = candles[entry_idx + d]
            if candle["low"] <= stop_loss: exit_price = stop_loss; break
            if candle["high"] >= target: exit_price = target; break
            exit_price = candle["close"]
        
        total_pnl += (exit_price - price) * shares
    return total_pnl

def run_experiment(exp_name, use_jitter=False, use_sector_exit=False):
    print(f"Running Experiment: {exp_name}...")
    spy_all = all_data["SPY"]["candles"]
    max_idx = len(spy_all) - SPRINT_DAYS - 1
    test_indices = [max_idx - (i * SPRINT_DAYS) for i in range(NUM_BACKTESTS)][::-1]
    
    capital = PORTFOLIO_START
    ticker_to_sector = {s["ticker"]: sector_mapping.get(s["sector"].split(' ')[0], "NONE") for s in web_data.get("all_stocks", [])}

    for idx in test_indices:
        # Regime Check (Simplified)
        s_spy = pd.Series([c["close"] for c in spy_all[:idx]])
        if s_spy.ewm(span=21, adjust=False).mean().iloc[-1] < s_spy.rolling(50).mean().iloc[-1]: continue
        
        leading_sectors = get_leading_sectors(idx, spy_all, all_data)
        
        candidates = []
        spy_bench = spy_all[idx]["close"] / spy_all[max(0, idx-63)]["close"]
        for t, d in all_data.items():
            if t == "SPY" or d.get("skip_calc", 1): continue
            cnd = d["candles"][:idx]
            if len(cnd) < 63: continue
            
            s_etf = ticker_to_sector.get(t, "NONE")
            if s_etf not in leading_sectors: continue
            
            # Flow / RMV filter
            c = cnd[-1]["close"]
            ranges = [o["high"] - o["low"] for o in cnd[-20:]]
            rmv = (sum(ranges)/len(ranges) / c) * 100 * 7.0
            avg_vol = sum([o["volume"] for o in cnd[-20:]]) / 20
            rvol = cnd[-1]["volume"] / avg_vol if avg_vol > 0 else 0
            
            if rvol > 1.2 and rmv < 40:
                rs_score = (c / cnd[-63]["close"]) / spy_bench
                candidates.append({"ticker": t, "rs": rs_score, "sector_etf": s_etf})
        
        candidates.sort(key=lambda x: -x["rs"])
        
        if use_jitter:
            # Take top 2 from leading sector, then next 3 from other sectors
            picks = []
            sectors_used = {}
            for c in candidates:
                sec = c["sector_etf"]
                if sectors_used.get(sec, 0) < 2: # Max 2 per sector
                    picks.append(c)
                    sectors_used[sec] = sectors_used.get(sec, 0) + 1
                if len(picks) >= NUM_PICKS: break
        else:
            picks = candidates[:NUM_PICKS]
            
        if not picks: continue
        capital += simulate_sprint(picks, capital, idx, use_sector_exit=use_sector_exit)
        
    total_return = ((capital - PORTFOLIO_START) / PORTFOLIO_START) * 100
    return {"name": exp_name, "return": total_return}

if __name__ == "__main__":
    e1 = run_experiment("CONTROL: Top 5 RS (All XLK likely)", use_jitter=False)
    e2 = run_experiment("EXPT 1: Sector Jitter (Max 2 per Sector)", use_jitter=True)
    e3 = run_experiment("EXPT 2: Sector RS Exit (9/21 Cross)", use_jitter=False, use_sector_exit=True)
    e4 = run_experiment("EXPT 3: Jitter + Sector RS Exit", use_jitter=True, use_sector_exit=True)
    
    print("\n" + "="*70)
    print(f"{'Experiment':<45} | {'Return %':<10}")
    print("-" * 70)
    for r in [e1, e2, e3, e4]:
        print(f"{r['name']:<45} | {r['return']:>8.1f}%")
    print("="*70)
