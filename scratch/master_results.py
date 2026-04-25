import json, os, sys, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, timedelta

ROOT = os.getcwd()
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")
all_data = json.load(open(PRICE_DATA))
web_data = json.load(open(WEB_DATA))
spy_all = all_data["SPY"]["candles"]

# Fetch VIX for Regime
vix_df = yf.download("^VIX", period="2y", progress=False)
if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
vix_df['sma20'] = vix_df['Close'].rolling(20).mean()

def run_sim(name, rmv_thr=40, sprint_days=15, use_vix=False, use_ema=False, use_jitter=False, use_trail=False):
    capital = 17000.0; wins = 0; losses = 0; total_pnl = 0
    max_idx = len(spy_all) - sprint_days - 1
    test_indices = [max_idx - (i * 20) for i in range(11)][::-1]
    
    for idx in test_indices:
        # Regime
        s_spy = pd.Series([c["close"] for c in spy_all[:idx]])
        ema9 = s_spy.ewm(span=9, adjust=False).mean().iloc[-1]
        sma21 = s_spy.rolling(21).mean().iloc[-1]
        dt = datetime.fromtimestamp(spy_all[idx]["datetime"]).strftime("%Y-%m-%d")
        vix_val = vix_df.loc[:dt]['Close'].iloc[-1] if dt in vix_df.index else 20
        vix_sma = vix_df.loc[:dt]['sma20'].iloc[-1] if dt in vix_df.index else 20
        
        if use_vix and (vix_val > 25 or vix_val > vix_sma): continue
        if use_ema and (ema9 < sma21): continue
        
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
        
        # Sprint Sim
        risk_per = (capital * 0.03) / 5
        for p in picks:
            cnd = all_data[p["t"]]["candles"]
            if idx >= len(cnd): continue
            pr = cnd[idx]["close"]
            st = pr * 0.95; tg = pr * 1.15; sh = int(risk_per / (pr*0.05))
            ex_pr = pr
            for d in range(1, sprint_days+1):
                if idx+d >= len(cnd): break
                c = cnd[idx+d]
                if c["low"] <= st: ex_pr = st; break
                if c["high"] >= tg: ex_pr = tg; break
                ex_pr = c["close"]
            pnl = (ex_pr - pr) * sh
            capital += pnl
            if pnl > 0: wins += 1
            else: losses += 1
            
    ret = ((capital - 17000)/17000)*100
    wr = (wins / (wins+losses)*100) if (wins+losses)>0 else 0
    return {"name": name, "wr": round(wr,1), "ret": round(ret,1)}

res = [
    run_sim("Base Alpha (Expt #4)", rmv_thr=40),
    run_sim("VIX + EMA Shield (Strongest)", use_vix=True, use_ema=True),
    run_sim("Tight VCP (RMV < 30)", rmv_thr=30),
    run_sim("Ultra Tight (RMV < 25)", rmv_thr=25),
    run_sim("4-Week Sprint", sprint_days=20),
    run_sim("8-Week Sprint", sprint_days=40),
    run_sim("Recommended: VIX + EMA + RMV30", use_vix=True, use_ema=True, rmv_thr=30)
]
print(json.dumps(res, indent=2))
