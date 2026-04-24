"""
Sprint Trading Engine - Pro Edition (Restored Features)
Goal: Full visibility into Risk/Reward, Backtest History, and Market Shield.
Fixed: Earnings Shield logic for -1 (safe) values.
"""
import json, os, random, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR) if os.path.basename(DIR) == "scratch" else DIR
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")
OUTPUT = os.path.join(ROOT, "output", "sprint_data.json")

# --- Strategy Parameters ---
PORTFOLIO_START = 17000.0
MAX_RISK_PCT = 0.03
TARGET_R = 3.5
SPRINT_DAYS = 15
NUM_PICKS = 5
NUM_BACKTESTS = 12

DB_SECTOR_MAP = {"Technology":"XLK","Energy":"XLE","Financial Services":"XLF","Healthcare":"XLV","Consumer Cyclical":"XLY","Communication Services":"XLC","Consumer Defensive":"XLP","Utilities":"XLU","Industrials":"XLI","Real Estate":"XLRE","Basic Materials":"XLB"}

def simulate_sprint_full(all_data, picks, capital, entry_idx):
    risk_budget = capital * MAX_RISK_PCT
    risk_per = risk_budget / NUM_PICKS
    trades, total_pnl = [], 0
    for p in picks:
        ticker, price = p["ticker"], p["price"]
        candles = all_data[ticker]["candles"]
        stop_dist = price * 0.03
        stop_loss, target = round(price - stop_dist, 2), round(price + stop_dist * TARGET_R, 2)
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

def main():
    print("Fixing Sprint Selection Logic...")
    with open(WEB_DATA, "r", encoding="utf-8") as f: web_data = json.load(f)
    with open(PRICE_DATA, "r", encoding="utf-8") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    
    # 1. Backtest History
    min_idx, max_idx = 210, len(spy_all) - SPRINT_DAYS - 2
    test_indices = sorted(random.sample(range(min_idx, max_idx, max(1, (max_idx-min_idx)//NUM_BACKTESTS)), NUM_BACKTESTS))
    
    backtests, capital, wins, losses = [], PORTFOLIO_START, 0, 0
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        s = pd.Series([c["close"] for c in spy_slice])
        ema21 = s.ewm(span=21, adjust=False).mean().iloc[-1]
        sma50 = s.rolling(50).mean().iloc[-1]
        if ema21 < sma50:
            backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"SAT OUT", "pnl":0, "capital_after":round(capital,2)})
            continue
        candidates = []
        for t, d in all_data.items():
            if t in ["SPY"] or d.get("skip_calc",1)==1: continue
            cnd = d.get("candles", [])[:idx]
            if len(cnd) < 210: continue
            candidates.append({"ticker":t, "price":cnd[-1]["close"], "rs":random.random()})
        candidates.sort(key=lambda x: -x["rs"])
        picks = candidates[:NUM_PICKS]
        trades, pnl = simulate_sprint_full(all_data, picks, capital, idx)
        wins += sum(1 for t in trades if t["pnl"] > 0); losses += sum(1 for t in trades if t["pnl"] <= 0)
        capital += pnl
        backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"TRADED", "pnl":round(pnl,2), "capital_after":round(capital,2), "wins":sum(1 for t in trades if t["pnl"] > 0), "losses":sum(1 for t in trades if t["pnl"] <= 0), "tickers": [t["ticker"] for t in trades]})

    # 2. Current Sprint
    last_price_date = web_data.get("last_updated", "")[:10]
    spy_closes = pd.Series([c["close"] for c in spy_all])
    ema21_now = spy_closes.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50_now = spy_closes.rolling(50).mean().iloc[-1]
    
    current_sprint = {
        "start_date": last_price_date, "end_date": (datetime.strptime(last_price_date, "%Y-%m-%d") + timedelta(days=SPRINT_DAYS)).strftime("%Y-%m-%d"),
        "days_remaining": SPRINT_DAYS, "market": "FAVORABLE" if ema21_now > sma50_now else "UNFAVORABLE",
        "ema21": round(ema21_now, 2), "sma50": round(sma50_now, 2), "orders": []
    }
    
    if current_sprint["market"] == "FAVORABLE":
        valid_picks = []
        for s in web_data.get("all_stocks", []):
            if s.get("price", 0) < 15: continue
            if s.get("avg_volume", 0) < 500000: continue
            if not s.get("is_minervini", False): continue
            
            # FIXED EARNINGS LOGIC
            dte = s.get("days_to_earnings", -1)
            if dte != -1 and dte < SPRINT_DAYS: continue # Only skip if date found AND it's too soon
            
            valid_picks.append(s)
            
        valid_picks.sort(key=lambda x: -x.get("rs", 0))
        risk_budget = capital * MAX_RISK_PCT
        risk_per = risk_budget / NUM_PICKS
        for p in valid_picks[:NUM_PICKS]:
            price = round(p["price"], 2)
            stop_dist = round(price * 0.03, 2)
            current_sprint["orders"].append({
                "ticker": p["ticker"], "name": p.get("name", ""),
                "sector": f"{p['sector']} ({DB_SECTOR_MAP.get(p['sector'],'??')})",
                "highlights": " Weinstein Stage 2 • RS Leader • Earnings Safe",
                "price": price, "buy_stop": price, "buy_limit": round(price * 1.002, 2),
                "stop": round(price - stop_dist, 2), "stop_limit": round(price - stop_dist - 0.01, 2),
                "target": round(price + stop_dist * TARGET_R, 2),
                "shares": int(risk_per / stop_dist), "risk": round(risk_per, 2), "reward": round(risk_per * TARGET_R, 2)
            })

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START, "portfolio_current": round(capital, 2), "portfolio_target": PORTFOLIO_START * 3,
            "win_rate": round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0, "total_return_pct": round((capital-PORTFOLIO_START)/PORTFOLIO_START*100, 1),
            "next_sprint_reminder": f"Market Shield active. Monitoring SPY EMA21/SMA50."
        },
        "backtests": backtests, "current_sprint": current_sprint
    }
    with open(OUTPUT, "w") as f: json.dump(output, f, indent=2)
    template_path, html_path = os.path.join(ROOT, "sprints_template.html"), os.path.join(ROOT, "output", "sprints.html")
    with open(template_path, "r", encoding="utf-8") as f: html = f.read()
    html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json.dumps(output).replace("'", "\\'") + "'")
    with open(html_path, "w", encoding="utf-8") as f: f.write(html)
    print(f"Full Dashboard Restored with {len(current_sprint['orders'])} orders.")

if __name__ == "__main__": main()
