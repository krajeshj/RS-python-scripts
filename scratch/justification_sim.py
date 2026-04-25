import json, os, sys, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timedelta

ROOT = os.getcwd()
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")
all_data = json.load(open(PRICE_DATA))
web_data = json.load(open(WEB_DATA))
spy_all = all_data["SPY"]["candles"]

# Fetch VIX
vix_df = yf.download("^VIX", period="2y", progress=False)
if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
vix_df['sma20'] = vix_df['Close'].rolling(20).mean()

def run_sim(rmv_thr=40, use_vix=True):
    capital = 17000.0; wins = 0; losses = 0; trades_count = 0
    max_idx = len(spy_all) - 22
    test_indices = [max_idx - (i * 20) for i in range(11)][::-1]
    
    for idx in test_indices:
        # 2. Market Shield (#2 VIX + #3 9/21 EMA)
        s_spy = pd.Series([c["close"] for c in spy_all[:idx]])
        ema9 = s_spy.ewm(span=9, adjust=False).mean().iloc[-1]
        sma21 = s_spy.rolling(21).mean().iloc[-1]
        
        dt = datetime.fromtimestamp(spy_all[idx]["datetime"]).strftime("%Y-%m-%d")
        vix_val = vix_df.loc[:dt]['Close'].iloc[-1] if dt in vix_df.index else 20
        vix_sma = vix_df.loc[:dt]['sma20'].iloc[-1] if dt in vix_df.index else 20
        
        # Shield logic: VIX < 25 AND VIX < SMA20 AND 9 EMA > 21 SMA
        shield_active = (vix_val < 25) and (vix_val < vix_sma) and (ema9 > sma21)
        if not shield_active: continue
        
        # Selection
        candidates = []
        spy_bench = spy_all[idx]["close"] / spy_all[max(0, idx-63)]["close"]
        for t, d in all_data.items():
            if t == "SPY" or d.get("skip_calc", 1): continue
            cnd = d["candles"][:idx]
            if len(cnd) < 63: continue
            c = cnd[-1]["close"]
            ranges = [o["high"] - o["low"] for o in cnd[-20:]]
            rmv = (sum(ranges)/20 / c) * 700
            if rmv < rmv_thr:
                rs = (c / cnd[-63]["close"]) / spy_bench
                candidates.append({"t": t, "rs": rs})
        
        candidates.sort(key=lambda x: -x["rs"])
        picks = candidates[:5]
        if not picks: continue
        
        risk_per = (capital * 0.03) / 5
        for p in picks:
            cnd = all_data[p["t"]]["candles"]
            if idx >= len(cnd): continue
            pr = cnd[idx]["close"]
            st = pr * 0.95; tg = pr * 1.15; sh = int(risk_per / (pr*0.05))
            ex_pr = pr
            for d in range(1, 16):
                if idx+d >= len(cnd): break
                c = cnd[idx+d]
                if c["low"] <= st: ex_pr = st; break
                if c["high"] >= tg: ex_pr = tg; break
                ex_pr = c["close"]
            pnl = (ex_pr - pr) * sh
            capital += pnl
            trades_count += 1
            if pnl > 0: wins += 1
            else: losses += 1
            
    return {"wr": round(wins/trades_count*100,1) if trades_count>0 else 0, "ret": round(((capital-17000)/17000)*100,1), "trades": trades_count}

res_a = run_sim(rmv_thr=40, use_vix=True) # VIX Only
res_b = run_sim(rmv_thr=30, use_vix=True) # VIX + RMV30

print(f"VARIANT A (VIX Shield Only): {res_a}")
print(f"VARIANT B (VIX Shield + RMV30): {res_b}")
