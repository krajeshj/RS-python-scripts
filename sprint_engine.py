"""
Sprint Trading Engine - Compounding & Validation Edition
Generates 11 historical backtests + current sprint recommendations.
Source: Full scan universe (web_data.json) for current, price_history for backtests.
Market Filter: SPY 21 EMA > 50 SMA.
Sector Filter: Only Leading/Improving sectors.
Strategy: 3R OCO on top RS stocks.
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
ETF_NAMES = {
    "XLK": "Technology", "XLE": "Energy", "XLF": "Financials", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLC": "Communication Services", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLI": "Industrials", "XLRE": "Real Estate", "XLB": "Materials"
}
SECTOR_MAP = {
    "Technology": "Technology", "Energy": "Energy", "Financials": "Financial Services",
    "Healthcare": "Healthcare", "Consumer Discretionary": "Consumer Cyclical",
    "Communication Services": "Communication Services", "Consumer Staples": "Consumer Defensive",
    "Utilities": "Utilities", "Industrials": "Industrials",
    "Real Estate": "Real Estate", "Materials": "Basic Materials"
}

def calc_ema(values, span):
    return pd.Series(values).ewm(span=span, adjust=False).mean()

def calc_sma(values, window):
    return pd.Series(values).rolling(window).mean()

def market_is_favorable(spy_candles):
    closes = [c["close"] for c in spy_candles]
    if len(closes) < 50: return False, 0, 0
    ema21 = calc_ema(closes, 21).iloc[-1]
    sma50 = calc_sma(closes, 50).iloc[-1]
    return ema21 > sma50, round(ema21, 2), round(sma50, 2)

def calc_rrg_quadrants(all_data, spy_candles):
    spy_closes = pd.Series([c["close"] for c in spy_candles])
    results = {}
    for pt in ETFS:
        t_candles = all_data.get(pt, {}).get("candles", [])
        if len(t_candles) < 30: continue
        try:
            min_len = min(len(t_candles), len(spy_closes))
            tc = pd.Series([c["close"] for c in t_candles[-min_len:]])
            sc = spy_closes.tail(min_len).reset_index(drop=True)
            rs_raw = tc / sc
            sma_r = rs_raw.rolling(14).mean()
            rs_ratio = 100 + ((sma_r - sma_r.rolling(14).mean()) / sma_r.rolling(14).std().replace(0, 1)) * 5
            roc = rs_ratio.diff(10)
            rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
            x, y = rs_ratio.iloc[-1] - 100, rs_mom.iloc[-1] - 100
            q = "Leading" if x > 0 and y > 0 else "Improving" if x < 0 and y > 0 else "Weakening" if x > 0 and y < 0 else "Lagging"
            results[ETF_NAMES[pt]] = q
        except: pass
    return results

def find_top_stocks(all_data, spy_candles, target_sectors, max_candle_idx):
    ref = pd.Series([c["close"] for c in spy_candles])
    candidates = []
    for ticker, data in all_data.items():
        if ticker == "SPY" or data.get("skip_calc", 1) == 1: continue
        if data.get("sector", "") not in target_sectors: continue
        candles = data.get("candles", [])[:max_candle_idx]
        if len(candles) < 200: continue
        closes = pd.Series([c["close"] for c in candles])
        price = candles[-1]["close"]
        # Simplified RS
        rs = (price / candles[-63]["close"]) / (ref.iloc[-1] / ref.iloc[-63])
        candidates.append({"ticker": ticker, "sector": data.get("sector", ""), "rs": rs, "price": price, "candles": candles})
    candidates.sort(key=lambda x: -x["rs"])
    return candidates[:NUM_PICKS]

def simulate_sprint(all_data, picks, capital, entry_idx):
    risk_budget = capital * MAX_RISK_PCT
    risk_per = risk_budget / len(picks) if picks else 0
    trades = []
    total_pnl = 0
    for p in picks:
        ticker = p["ticker"]
        price = p["price"]
        candles = all_data[ticker]["candles"]
        # Simplified stop (3%)
        stop_dist = price * 0.03
        stop_loss = round(price - stop_dist, 2)
        target = round(price + stop_dist * TARGET_R, 2)
        shares = max(1, int(risk_per / stop_dist))
        
        exit_price = price
        result = "EXPIRED"
        for d in range(1, SPRINT_DAYS + 1):
            if entry_idx + d >= len(candles): break
            low, high = candles[entry_idx+d]["low"], candles[entry_idx+d]["high"]
            if low <= stop_loss: exit_price, result = stop_loss, "STOP"; break
            if high >= target: exit_price, result = target, "TARGET"; break
            exit_price = candles[entry_idx+d]["close"]
        
        pnl = (exit_price - price) * shares
        total_pnl += pnl
        trades.append({"ticker": ticker, "pnl": pnl, "result": result})
    return trades, total_pnl

def main():
    print("Loading data for validation...")
    with open(PRICE_DATA, "r") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    
    # --- Backtest ---
    min_idx, max_idx = 205, len(spy_all) - SPRINT_DAYS - 2
    step = (max_idx - min_idx) // (NUM_BACKTESTS + 1)
    test_indices = sorted(random.sample(range(min_idx, max_idx, max(1, step)), NUM_BACKTESTS))
    
    backtests, capital, wins, losses, sat_out = [], PORTFOLIO_START, 0, 0, 0
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        date_str = datetime.fromtimestamp(spy_all[idx]["datetime"], tz=timezone.utc).strftime("%Y-%m-%d")
        favorable, ema21, sma50 = market_is_favorable(spy_slice)
        if not favorable:
            sat_out += 1
            backtests.append({"sprint": len(backtests)+1, "date": date_str, "action": "SAT OUT", "pnl": 0, "capital_after": capital, "ema21": ema21, "sma50": sma50})
            continue
        rrg = calc_rrg_quadrants(all_data, spy_slice)
        target_sectors = {SECTOR_MAP[s] for s, q in rrg.items() if q in ("Leading", "Improving") and s in SECTOR_MAP}
        picks = find_top_stocks(all_data, spy_slice, target_sectors, idx)
        if not picks:
            sat_out += 1
            backtests.append({"sprint": len(backtests)+1, "date": date_str, "action": "NO PICKS", "pnl": 0, "capital_after": capital, "ema21": ema21, "sma50": sma50})
            continue
        trades, pnl = simulate_sprint(all_data, picks, capital, idx)
        wins += sum(1 for t in trades if t["pnl"] > 0)
        losses += sum(1 for t in trades if t["pnl"] <= 0)
        capital += pnl
        backtests.append({"sprint": len(backtests)+1, "date": date_str, "action": "TRADED", "pnl": pnl, "capital_after": capital, "wins": sum(1 for t in trades if t["pnl"] > 0), "losses": sum(1 for t in trades if t["pnl"] <= 0), "sectors": list(target_sectors)})

    # --- Current Sprint ---
    print("Generating current recommendations...")
    with open(WEB_DATA, "r") as f: web_data = json.load(f)
    favorable, ema21_now, sma50_now = market_is_favorable(spy_all)
    rrg_now = calc_rrg_quadrants(all_data, spy_all)
    winning_sectors = {s for s, q in rrg_now.items() if q in ("Leading", "Improving")}
    target_sectors = {SECTOR_MAP[s] for s in winning_sectors if s in SECTOR_MAP}
    
    current_sprint = {"date": web_data.get("last_updated", "")[:10], "market": "FAVORABLE" if favorable else "UNFAVORABLE", "ema21": ema21_now, "sma50": sma50_now, "action": "ENTER NEW SPRINT" if favorable else "SIT OUT", "orders": []}
    if favorable:
        candidates = [s for s in web_data.get("all_stocks", []) if s.get("sector") in target_sectors]
        candidates.sort(key=lambda x: -x.get("rs", 0))
        risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
        for p in candidates[:NUM_PICKS]:
            price = p["price"]
            stop_dist = price * 0.03 # Standard 3%
            stop_loss = round(price - stop_dist, 2)
            current_sprint["orders"].append({
                "ticker": p["ticker"], "name": p.get("name", ""), "sector": p.get("sector", ""), "price": price,
                "buy_stop": price, "buy_limit": round(price * 1.002, 2),
                "stop": stop_loss, "stop_limit": round(stop_loss * 0.998, 2),
                "target": round(price + stop_dist * TARGET_R, 2),
                "shares": max(1, int(risk_per / stop_dist)),
                "sl_pct": 3.0, "target_pct": 9.0, "risk": round(risk_per, 2), "potential": round(risk_per * TARGET_R, 2)
            })

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START, "portfolio_current": round(capital, 2), "portfolio_target": PORTFOLIO_START * 3,
            "target_r": TARGET_R, "win_rate": round(wins/(wins+losses)*100, 1) if (wins+losses)>0 else 0,
            "total_return_pct": round((capital - PORTFOLIO_START) / PORTFOLIO_START * 100, 1),
            "traded": len([b for b in backtests if b["action"] == "TRADED"]), "total_backtests": NUM_BACKTESTS, "sat_out": sat_out,
            "next_sprint_reminder": "Cycle every 15 trading days."
        },
        "backtests": backtests, "current_sprint": current_sprint
    }
    with open(OUTPUT, "w") as f: json.dump(output, f, indent=2)
    
    # Update HTML
    template_path, html_path = os.path.join(ROOT, "sprints_template.html"), os.path.join(ROOT, "output", "sprints.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f: html = f.read()
        html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json.dumps(output).replace("'", "\\'") + "'")
        with open(html_path, "w", encoding="utf-8") as f: f.write(html)
        print("Sprints updated with backtest validation.")

if __name__ == "__main__": main()
