"""
Sprint Trading Engine - Compounding Edition
Triggered after daily scan. Uses full scan universe + Sector Rotation filtering.
Market Filter: SPY 21 EMA > 50 SMA.
Sector Filter: Only Leading/Improving sectors.
Strategy: Top 5 RS stocks in winning sectors.
"""
import json, os, sys
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
NUM_PICKS = 5

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

def get_market_status(all_data):
    spy = all_data.get("SPY", {}).get("candles", [])
    if len(spy) < 50: return False, 0, 0
    closes = [c["close"] for c in spy]
    ema21 = calc_ema(closes, 21).iloc[-1]
    sma50 = calc_sma(closes, 50).iloc[-1]
    return ema21 > sma50, round(ema21, 2), round(sma50, 2)

def main():
    if not os.path.exists(PRICE_DATA) or not os.path.exists(WEB_DATA):
        print("Data files missing. Run daily scan first.")
        return

    with open(PRICE_DATA, "r") as f:
        all_price_data = json.load(f)
    with open(WEB_DATA, "r") as f:
        web_data = json.load(f)

    favorable, ema21, sma50 = get_market_status(all_price_data)
    rrg = calc_rrg_quadrants(all_price_data, all_price_data["SPY"]["candles"])
    winning_sectors = {s for s, q in rrg.items() if q in ("Leading", "Improving")}
    # Map to the sector names used in the stocks database
    target_sectors = {SECTOR_MAP[s] for s in winning_sectors if s in SECTOR_MAP}

    current_sprint = {
        "date": web_data.get("last_updated", "")[:10],
        "market": "FAVORABLE" if favorable else "UNFAVORABLE",
        "ema21": ema21, "sma50": sma50,
        "winning_sectors": list(winning_sectors),
        "orders": []
    }

    if favorable:
        all_stocks = web_data.get("all_stocks", [])
        # Filter for winning sectors
        candidates = [s for s in all_stocks if s.get("sector") in target_sectors]
        # Rank by RS
        candidates.sort(key=lambda x: -x.get("rs", 0))
        picks = candidates[:NUM_PICKS]

        risk_budget = PORTFOLIO_START * MAX_RISK_PCT
        risk_per = risk_budget / len(picks) if picks else 0

        for p in picks:
            ticker = p["ticker"]
            price = p["price"]
            candles = all_price_data.get(ticker, {}).get("candles", [])
            
            # Simple ATR-like stop: 1.5x of 10-day volatility or 3%
            if len(candles) >= 10:
                vols = [abs(c['high']-c['low']) for c in candles[-10:]]
                stop_dist = max(np.mean(vols) * 1.5, price * 0.02)
            else:
                stop_dist = price * 0.03
            
            stop_loss = round(price - stop_dist, 2)
            target = round(price + stop_dist * TARGET_R, 2)
            shares = max(1, int(risk_per / stop_dist))

            current_sprint["orders"].append({
                "ticker": ticker,
                "name": p.get("name", ticker),
                "sector": p.get("sector", "unknown"),
                "price": price,
                "buy_stop": price,
                "buy_limit": round(price * 1.002, 2), # 0.2% offset
                "stop": stop_loss,
                "stop_limit": round(stop_loss * 0.998, 2), # 0.2% offset
                "target": target,
                "shares": shares,
                "sl_pct": round((stop_dist / price) * 100, 1),
                "target_pct": round((stop_dist * TARGET_R / price) * 100, 1),
                "risk": round(shares * stop_dist, 2),
                "potential": round(shares * stop_dist * TARGET_R, 2)
            })

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START,
            "portfolio_current": PORTFOLIO_START,
            "portfolio_target": PORTFOLIO_START * 3,
            "target_r": TARGET_R,
            "next_sprint_reminder": "Cycle every 15 trading days. Review next cycle in 15 days."
        },
        "current_sprint": current_sprint,
        "backtests": [] # Backtest data from previous logic can be integrated or run separately
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    
    # Update HTML
    template_path = os.path.join(ROOT, "sprints_template.html")
    html_path = os.path.join(ROOT, "output", "sprints.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
        json_str = json.dumps(output).replace("'", "\\'")
        html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json_str + "'")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Sprints updated successfully.")

if __name__ == "__main__":
    main()
