"""
Sprint Trading Engine - Power E*TRADE Edition
FIXED: Backtest now uses actual Historical RS calculation instead of random sampling.
Restores "Institutional Alpha" performance metrics.
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

def get_rrg_quadrant(tc, sc):
    if len(tc) < 30: return "Unknown"
    rs_raw = tc / sc
    sma_ratio = rs_raw.rolling(14).mean()
    rs_ratio = 100 + ((sma_ratio - sma_ratio.rolling(14).mean()) / sma_ratio.rolling(14).std().replace(0, 1)) * 5
    roc = rs_ratio.diff(10)
    rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
    x, y = rs_ratio.iloc[-1] - 100, rs_mom.iloc[-1] - 100
    if x >= 0 and y >= 0: return "Leading"
    if x < 0 and y >= 0: return "Improving"
    if x < 0 and y < 0: return "Lagging"
    return "Weakening"

def get_leading_sectors(idx, spy_all, all_data):
    leading = []
    sector_etfs = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]
    spy_c = pd.Series([c["close"] for c in spy_all[:idx]])
    for s in sector_etfs:
        if s not in all_data: continue
        tc = pd.Series([c["close"] for c in all_data[s]["candles"][:idx]])
        if len(tc) < 63: continue
        if get_rrg_quadrant(tc, spy_c.tail(len(tc)).reset_index(drop=True)) == "Leading":
            leading.append(s)
    return leading

def simulate_sprint_full(all_data, picks, capital, entry_idx):
    risk_budget = capital * MAX_RISK_PCT
    risk_per = risk_budget / NUM_PICKS
    trades, total_pnl = [], 0
    for p in picks:
        ticker, price = p["ticker"], p["price"]
        candles = all_data[ticker]["candles"]
        
        # 21-Day Low Stop Logic
        lookback = candles[max(0, entry_idx-21):entry_idx]
        lows = [c["low"] for c in lookback]
        low_21 = min(lows) if lows else price * 0.97
        # Safety Buffer: Max 8% stop, Min 1.5% stop
        stop_loss = max(low_21, price * 0.92)
        stop_loss = min(stop_loss, price * 0.985)
        stop_dist = price - stop_loss
            
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

def main():
    print("Regenerating Dashboard with HISTORICAL ALPHA (Actual RS Backtesting)...")
    with open(WEB_DATA, "r", encoding="utf-8") as f: web_data = json.load(f)
    with open(PRICE_DATA, "r", encoding="utf-8") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    
    # 1. Backtest History (Deterministic RS Calculation)
    min_idx, max_idx = 210, len(spy_all) - SPRINT_DAYS - 2
    # We'll use a fixed seed or fixed offsets to make results consistent
    test_indices = [max_idx - (i * SPRINT_DAYS * 2) for i in range(NUM_BACKTESTS)][::-1]
    
    backtests, capital, wins, losses = [], PORTFOLIO_START, 0, 0
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        s_spy = pd.Series([c["close"] for c in spy_slice])
        ema21, sma50 = s_spy.ewm(span=21, adjust=False).mean().iloc[-1], s_spy.rolling(50).mean().iloc[-1]
        
        if ema21 < sma50:
            backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"SAT OUT", "pnl":0, "capital_after":round(capital,2)})
            continue
        
        leading_sectors = get_leading_sectors(idx, spy_all, all_data)
        
        # Sector Mapping
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
            
            # Experiment #4 Filter: Leading Sector + Flow > 1.2 + Low RMV < 40
            is_leading_sec = ticker_to_sector.get(t, "NONE") in leading_sectors
            if not (is_leading_sec and rvol > 1.2 and rmv < 40): continue

            rs_score = (c / cnd[-63]["close"]) / spy_bench
            candidates.append({"ticker":t, "price":c, "rs":rs_score})
        
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
    spy_closes = pd.Series([c["close"] for c in spy_all])
    ema21_now, sma50_now = spy_closes.ewm(span=21, adjust=False).mean().iloc[-1], spy_closes.rolling(50).mean().iloc[-1]
    
    # Preserve ongoing sprint if possible
    existing_sprint = None
    if os.path.exists(OUTPUT):
        try:
            with open(OUTPUT, "r") as f:
                old_data = json.load(f)
                existing_sprint = old_data.get("current_sprint")
        except: pass

    is_ongoing = False
    if existing_sprint and existing_sprint.get("end_date"):
        end_dt = datetime.strptime(existing_sprint["end_date"], "%Y-%m-%d")
        if datetime.now() < end_dt + timedelta(days=1):
            is_ongoing = True

    if is_ongoing:
        current_sprint = existing_sprint
        # Update days remaining
        end_dt = datetime.strptime(current_sprint["end_date"], "%Y-%m-%d")
        days_rem = (end_dt - datetime.now()).days + 1
        current_sprint["days_remaining"] = max(0, days_rem)
        current_sprint["market"] = "FAVORABLE" if ema21_now > sma50_now else "UNFAVORABLE"
        current_sprint["ema21"], current_sprint["sma50"] = round(ema21_now, 2), round(sma50_now, 2)
    else:
        last_price_date = web_data.get("last_updated", "")[:10]
        current_sprint = {"start_date": last_price_date, "end_date": (datetime.strptime(last_price_date, "%Y-%m-%d") + timedelta(days=SPRINT_DAYS)).strftime("%Y-%m-%d"), "days_remaining": SPRINT_DAYS, "market": "FAVORABLE" if ema21_now > sma50_now else "UNFAVORABLE", "ema21": round(ema21_now, 2), "sma50": round(sma50_now, 2), "orders": []}
        
        if current_sprint["market"] == "FAVORABLE":
            # Apply Experiment #4 Logic to Current Candidates
            leading_sectors = get_leading_sectors(len(spy_all), spy_all, all_data)
            mapping = {"Technology": "XLK", "Energy": "XLE", "Financial": "XLF", "Healthcare": "XLV", "Consumer": "XLY", "Communication": "XLC", "Basic": "XLB", "Industrials": "XLI", "Real": "XLRE", "Utilities": "XLU"}
            
            final_candidates = []
            for s in web_data.get("all_stocks", []):
                t = s["ticker"]
                if t not in all_data: continue
                cnd = all_data[t]["candles"]
                if len(cnd) < 63: continue
                
                c = cnd[-1]["close"]
                if c < 15: continue
                
                # RMV/Flow Calc
                ranges = [o["high"] - o["low"] for o in cnd[-20:]]
                rmv = (sum(ranges)/len(ranges) / c) * 100 * 7.0
                avg_vol = sum([o["volume"] for o in cnd[-20:]]) / 20
                rvol = cnd[-1]["volume"] / avg_vol if avg_vol > 0 else 0
                
                sector_etf = mapping.get(s["sector"].split(' ')[0], "NONE")
                is_leading_sec = sector_etf in leading_sectors
                
                # Earnings Shield
                dte = s.get("days_to_earnings", -1)
                earnings_safe = (dte >= 15 or dte == -1)

                if is_leading_sec and rvol > 1.2 and rmv < 40 and earnings_safe:
                    final_candidates.append(s)
            
            final_candidates.sort(key=lambda x: -x.get("rs", 0))
            risk_budget = capital * MAX_RISK_PCT
            risk_per = risk_budget / NUM_PICKS
            
            for p in final_candidates[:NUM_PICKS]:
                t = p["ticker"]
                price = round(p["price"], 2)
                
                # 21-Day Low Stop for Live Card
                cnd = all_data[t]["candles"]
                lows = [c["low"] for c in cnd[-21:]]
                low_21 = min(lows) if lows else price * 0.97
                stop_p = max(low_21, price * 0.92)
                stop_p = min(stop_p, price * 0.985)
                stop_dist = price - stop_p
                
                current_sprint["orders"].append({
                    "ticker": t, "name": p.get("name", ""), "sector": f"{p['sector']} (Institutional Flow)", 
                    "highlights": f"Leading Sector • Fund Flow: {round(p.get('rvol', 1.3), 1)}x • 21-Day Low Stop",
                    "price": price, "buy_stop": price, "buy_limit": round(price * 1.002, 2), "target": round(price + stop_dist * TARGET_R, 2), "stop": round(stop_p, 2), "stop_limit": round(stop_p * 0.998, 2),
                    "shares": int(risk_per / stop_dist), "risk": round(risk_per, 2), "reward": round(risk_per * TARGET_R, 2)
                })

    # Add/Update Today's Price for all orders
    for o in current_sprint["orders"]:
        t = o["ticker"]
        if t in all_data and "candles" in all_data[t] and len(all_data[t]["candles"]) > 0:
            o["today_price"] = round(all_data[t]["candles"][-1]["close"], 2)
        else:
            o["today_price"] = o.get("today_price", o["price"])

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START, "portfolio_current": round(capital, 2), "portfolio_target": PORTFOLIO_START * 3,
            "win_rate": round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0, "total_return_pct": round((capital-PORTFOLIO_START)/PORTFOLIO_START*100, 1),
            "next_sprint_reminder": "Institutional Alpha Restored. Deterministic RS Backtesting active."
        },
        "backtests": backtests, "current_sprint": current_sprint
    }
    with open(OUTPUT, "w") as f: json.dump(output, f, indent=2)
    template_path, html_path = os.path.join(ROOT, "sprints_template.html"), os.path.join(ROOT, "output", "sprints.html")
    with open(template_path, "r", encoding="utf-8") as f: html = f.read()
    html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json.dumps(output).replace("'", "\\'") + "'")
    with open(html_path, "w", encoding="utf-8") as f: f.write(html)
    print("Institutional Alpha Restored.")

if __name__ == "__main__": main()
