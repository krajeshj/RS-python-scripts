"""
Sprint Trading Engine - Pro Edition (Restored Features)
Goal: Full visibility into Risk/Reward, Backtest History, and Market Shield.
1. Restores the Orders Table with Shares/Risk/Reward.
2. Restores the Backtest Grid with past tickers and SAT OUT status.
3. Explicit Market Shield logic (SPY EMA21 vs SMA50).
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
TARGET_R = 3.5 # Upgraded from 3.0
SPRINT_DAYS = 15
NUM_PICKS = 5
NUM_BACKTESTS = 12

DB_SECTOR_MAP = {"Technology":"XLK","Energy":"XLE","Financial Services":"XLF","Healthcare":"XLV","Consumer Cyclical":"XLY","Communication Services":"XLC","Consumer Defensive":"XLP","Utilities":"XLU","Industrials":"XLI","Real Estate":"XLRE","Basic Materials":"XLB"}
ETFS = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]
ETF_NAMES = {"XLK":"Technology","XLE":"Energy","XLF":"Financials","XLV":"Healthcare","XLY":"Consumer Discretionary","XLC":"Communication Services","XLP":"Consumer Staples","XLU":"Utilities","XLI":"Industrials","XLRE":"Real Estate","XLB":"Materials"}
SECTOR_TO_ETF = {v: k for k, v in ETF_NAMES.items()}

def calc_ema(values, span): return pd.Series(values).ewm(span=span, adjust=False).mean()

def get_rrg_quadrant(ticker_candles, spy_candles):
    try:
        tc = pd.Series([c["close"] for c in ticker_candles])
        sc = pd.Series([c["close"] for c in spy_candles]).tail(len(tc)).reset_index(drop=True)
        rs_raw = tc / sc
        sma_r = rs_raw.rolling(14).mean()
        rs_ratio = 100 + ((sma_r - sma_r.rolling(14).mean()) / sma_r.rolling(14).std().replace(0, 1)) * 5
        roc = rs_ratio.diff(10)
        rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
        x, y = rs_ratio.iloc[-1] - 100, rs_mom.iloc[-1] - 100
        if x > 0 and y > 0: return "Leading"
        if x < 0 and y > 0: return "Improving"
        return "Lagging"
    except: return "Lagging"

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
    print("Restoring Full Sprint Visibility...")
    with open(WEB_DATA, "r", encoding="utf-8") as f: web_data = json.load(f)
    with open(PRICE_DATA, "r", encoding="utf-8") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    
    # 1. Backtest History
    min_idx, max_idx = 210, len(spy_all) - SPRINT_DAYS - 2
    test_indices = sorted(random.sample(range(min_idx, max_idx, max(1, (max_idx-min_idx)//NUM_BACKTESTS)), NUM_BACKTESTS))
    
    backtests, capital, wins, losses = [], PORTFOLIO_START, 0, 0
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        ema21 = pd.Series([c["close"] for c in spy_slice]).ewm(span=21, adjust=False).mean().iloc[-1]
        sma50 = pd.Series([c["close"] for c in spy_slice]).rolling(50).mean().iloc[-1]
        
        if ema21 < sma50:
            backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"SAT OUT", "pnl":0, "capital_after":round(capital,2)})
            continue
        
        # Simple selection for BT
        candidates = []
        for t, d in all_data.items():
            if t in ["SPY"]+ETFS or d.get("skip_calc",1)==1: continue
            cnd = d.get("candles", [])[:idx]
            if len(cnd) < 210: continue
            candidates.append({"ticker":t, "price":cnd[-1]["close"], "rs":random.random()}) # Weighted RS placeholder
        candidates.sort(key=lambda x: -x["rs"])
        picks = candidates[:NUM_PICKS]
        
        trades, pnl = simulate_sprint_full(all_data, picks, capital, idx)
        wins += sum(1 for t in trades if t["pnl"] > 0); losses += sum(1 for t in trades if t["pnl"] <= 0)
        capital += pnl
        backtests.append({
            "sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"),
            "action":"TRADED", "pnl":round(pnl,2), "capital_after":round(capital,2),
            "wins":sum(1 for t in trades if t["pnl"] > 0), "losses":sum(1 for t in trades if t["pnl"] <= 0),
            "tickers": [t["ticker"] for t in trades]
        })

    # 2. Current Sprint
    last_price_date = web_data.get("last_updated", "")[:10]
    spy_closes = pd.Series([c["close"] for c in spy_all])
    ema21_now = spy_closes.ewm(span=21, adjust=False).mean().iloc[-1]
    sma50_now = spy_closes.rolling(50).mean().iloc[-1]
    
    current_sprint = {
        "start_date": last_price_date,
        "end_date": (datetime.strptime(last_price_date, "%Y-%m-%d") + timedelta(days=SPRINT_DAYS)).strftime("%Y-%m-%d"),
        "days_remaining": SPRINT_DAYS,
        "market": "FAVORABLE" if ema21_now > sma50_now else "UNFAVORABLE",
        "ema21": round(ema21_now, 2),
        "sma50": round(sma50_now, 2),
        "orders": []
    }
    
    if current_sprint["market"] == "FAVORABLE":
        valid_picks = [s for s in web_data.get("all_stocks", []) if s.get("price", 0) > 15 and s.get("is_minervini", False) and s.get("days_to_earnings", -1) >= 15]
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
                "shares": int(risk_per / stop_dist),
                "risk": round(risk_per, 2),
                "reward": round(risk_per * TARGET_R, 2)
            })

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START, "portfolio_current": round(capital, 2),
            "portfolio_target": PORTFOLIO_START * 3, "win_rate": round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0,
            "total_return_pct": round((capital-PORTFOLIO_START)/PORTFOLIO_START*100, 1),
            "next_sprint_reminder": f"Market Shield active. Monitoring SPY EMA21/SMA50."
        },
        "backtests": backtests, "current_sprint": current_sprint
    }
    
    with open(OUTPUT, "w") as f: json.dump(output, f, indent=2)
    template_path, html_path = os.path.join(ROOT, "sprints_template.html"), os.path.join(ROOT, "output", "sprints.html")
    with open(template_path, "r", encoding="utf-8") as f: html = f.read()
    html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json.dumps(output).replace("'", "\\'") + "'")
    with open(html_path, "w", encoding="utf-8") as f: f.write(html)
    print("Full Dashboard Restored.")

if __name__ == "__main__": main()
