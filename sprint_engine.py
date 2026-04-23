"""
Sprint Trading Engine - High Conviction Edition
Tweaked for higher win-rate and returns via:
1. Weighted Relative Strength (40/20/20/20)
2. Immediate Momentum Filter (Price > 10 EMA)
3. Volume Confirmation (> 90% of 50-day average)
4. ATR-Adjusted Stops (Adaptive Risk)
"""
import json, os, random, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR) if os.path.basename(DIR) == "scratch" else DIR
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")
OUTPUT = os.path.join(ROOT, "output", "sprint_data.json")

# --- Strategy Parameters ---
PORTFOLIO_START = 17000.0
MAX_RISK_PCT = 0.03
TARGET_R = 3
SPRINT_DAYS = 15
NUM_PICKS = 5
NUM_BACKTESTS = 11

# --- Sector Mapping ---
ETFS = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]
ETF_NAMES = {"XLK":"Technology","XLE":"Energy","XLF":"Financials","XLV":"Healthcare","XLY":"Consumer Discretionary","XLC":"Communication Services","XLP":"Consumer Staples","XLU":"Utilities","XLI":"Industrials","XLRE":"Real Estate","XLB":"Materials"}
SECTOR_MAP = {"Technology":"Technology","Energy":"Energy","Financials":"Financial Services","Healthcare":"Healthcare","Consumer Discretionary":"Consumer Cyclical","Communication Services":"Communication Services","Consumer Staples":"Consumer Defensive","Utilities":"Utilities","Industrials":"Industrials","Real Estate":"Real Estate","Materials":"Basic Materials"}

def calc_ema(values, span):
    return pd.Series(values).ewm(span=span, adjust=False).mean()

def calc_atr(candles, n=14):
    if len(candles) < n+1: return None
    trs = [max(c['high']-c['low'], abs(c['high']-candles[i-1]['close']), abs(c['low']-candles[i-1]['close'])) for i, c in enumerate(candles) if i > 0]
    return np.mean(trs[-n:])

def weighted_rs(closes, ref):
    def qr(n):
        try:
            l = min(len(closes), n*63)
            return (closes.iloc[-1]/closes.iloc[-l]) / (ref.iloc[-1]/ref.iloc[-l])
        except: return 0
    return 0.4*qr(1) + 0.2*qr(2) + 0.2*qr(3) + 0.2*qr(4)

def find_top_stocks(all_data, spy_candles, target_sectors, max_candle_idx):
    ref = pd.Series([c["close"] for c in spy_candles])
    candidates = []
    for ticker, data in all_data.items():
        if ticker=="SPY" or data.get("skip_calc",1)==1: continue
        if data.get("sector","") not in target_sectors: continue
        candles = data.get("candles", [])[:max_candle_idx]
        if len(candles) < 200: continue
        
        closes = pd.Series([c["close"] for c in candles])
        price = candles[-1]["close"]
        ema10 = closes.ewm(span=10, adjust=False).mean().iloc[-1]
        
        # Tweak 1: Price > 10 EMA (Immediate Momentum)
        if price < ema10: continue
        
        # Tweak 2: Volume > 90% of 50-day Avg
        avg_vol = np.mean([c["volume"] for c in candles[-50:]])
        if candles[-1]["volume"] < avg_vol * 0.9: continue
        
        rs = weighted_rs(closes, ref)
        atr = calc_atr(candles)
        candidates.append({"ticker":ticker, "sector":data.get("sector",""), "rs":rs, "price":price, "atr":atr, "highlights":data.get("highlights","")})
    
    candidates.sort(key=lambda x: -x["rs"])
    return candidates[:NUM_PICKS]

def simulate_sprint(all_data, picks, capital, entry_idx):
    risk_budget = capital * MAX_RISK_PCT
    risk_per = risk_budget / len(picks) if picks else 0
    trades = []
    total_pnl = 0
    for p in picks:
        ticker, price, atr = p["ticker"], p["price"], p["atr"]
        candles = all_data[ticker]["candles"]
        # Tweak 3: ATR-Based Stop (1.5x ATR or min 2%)
        stop_dist = max(atr * 1.5, price * 0.02)
        stop_loss = round(price - stop_dist, 2)
        target = round(price + stop_dist * TARGET_R, 2)
        shares = max(1, int(risk_per / stop_dist))
        
        exit_price, result = price, "EXPIRED"
        for d in range(1, SPRINT_DAYS + 1):
            if entry_idx + d >= len(candles): break
            l, h = candles[entry_idx+d]["low"], candles[entry_idx+d]["high"]
            if l <= stop_loss: exit_price, result = stop_loss, "STOP"; break
            if h >= target: exit_price, result = target, "TARGET"; break
            exit_price = candles[entry_idx+d]["close"]
        
        pnl = (exit_price - price) * shares
        total_pnl += pnl
        trades.append({"ticker":ticker, "pnl":pnl, "result":result})
    return trades, total_pnl

def main():
    with open(PRICE_DATA, "r") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    
    # Backtest
    min_idx, max_idx = 205, len(spy_all) - SPRINT_DAYS - 2
    test_indices = sorted(random.sample(range(min_idx, max_idx, max(1, (max_idx-min_idx)//12)), NUM_BACKTESTS))
    
    backtests, capital, wins, losses, sat_out = [], PORTFOLIO_START, 0, 0, 0
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        ema21 = calc_ema([c["close"] for c in spy_slice], 21).iloc[-1]
        sma50 = pd.Series([c["close"] for c in spy_slice]).rolling(50).mean().iloc[-1]
        
        if ema21 < sma50:
            sat_out += 1
            backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"SAT OUT", "pnl":0, "capital_after":capital, "ema21":round(ema21,2), "sma50":round(sma50,2)})
            continue
            
        rrg = {ETF_NAMES[pt]: "Leading" for pt in ETFS} # Simplified for speed in backtest loop, can be expanded
        target_sectors = {SECTOR_MAP[s] for s in ETF_NAMES.values() if s in SECTOR_MAP}
        picks = find_top_stocks(all_data, spy_slice, target_sectors, idx)
        
        trades, pnl = simulate_sprint(all_data, picks, capital, idx)
        wins += sum(1 for t in trades if t["pnl"] > 0)
        losses += sum(1 for t in trades if t["pnl"] <= 0)
        capital += pnl
        backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"TRADED", "pnl":pnl, "capital_after":capital, "wins":sum(1 for t in trades if t["pnl"] > 0), "losses":sum(1 for t in trades if t["pnl"] <= 0)})

    # Current
    with open(WEB_DATA, "r") as f: web_data = json.load(f)
    closes_now = [c["close"] for c in spy_all]
    ema21_now, sma50_now = calc_ema(closes_now, 21).iloc[-1], pd.Series(closes_now).rolling(50).mean().iloc[-1]
    
    current_sprint = {"date":web_data.get("last_updated","")[:10], "market":"FAVORABLE" if ema21_now > sma50_now else "UNFAVORABLE", "ema21":round(ema21_now,2), "sma50":round(sma50_now,2), "orders":[]}
    if ema21_now > sma50_now:
        # Use full universe
        candidates = [s for s in web_data.get("all_stocks", [])]
        # Weighted RS ranking
        candidates.sort(key=lambda x: -x.get("rs", 0))
        risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
        for p in candidates[:NUM_PICKS]:
            price, ticker = p["price"], p["ticker"]
            candles = all_data.get(ticker,{}).get("candles",[])
            atr = calc_atr(candles) if candles else price * 0.03
            stop_dist = max(atr * 1.5, price * 0.02)
            stop_loss = round(price - stop_dist, 2)
            current_sprint["orders"].append({
                "ticker":ticker, "name":p.get("name",""), "sector":p.get("sector",""), "price":price,
                "highlights":p.get("highlights",""), "buy_stop":price, "buy_limit":round(price*1.002,2),
                "stop":stop_loss, "stop_limit":round(stop_loss*0.998,2), "target":round(price+stop_dist*TARGET_R,2),
                "shares":max(1, int(risk_per/stop_dist)), "sl_pct":round((stop_dist/price)*100,1), "target_pct":round((stop_dist*TARGET_R/price)*100,1),
                "risk":round(risk_per,2), "potential":round(risk_per*TARGET_R,2)
            })

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START, "portfolio_current": round(capital,2), "portfolio_target": PORTFOLIO_START*3,
            "win_rate": round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0, "total_return_pct": round((capital-PORTFOLIO_START)/PORTFOLIO_START*100,1),
            "next_sprint_reminder": "Cycle every 15 days."
        },
        "backtests": backtests, "current_sprint": current_sprint
    }
    with open(OUTPUT, "w") as f: json.dump(output, f, indent=2)
    template_path, html_path = os.path.join(ROOT, "sprints_template.html"), os.path.join(ROOT, "output", "sprints.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f: html = f.read()
        html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json.dumps(output).replace("'", "\\'") + "'")
        with open(html_path, "w", encoding="utf-8") as f: f.write(html)
        print("Sprints optimized for conviction.")

if __name__ == "__main__": main()
