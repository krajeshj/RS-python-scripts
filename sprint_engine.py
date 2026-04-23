"""
Sprint Trading Engine - Rajesh_Alpha Final
Replicating the high-performing 9.5% return logic.
1. 15-Day Sprints.
2. 3R Target / 3% Flat Stop.
3. 5 Equal-Weighted Picks.
4. Sector RRG: Leading/Improving only.
5. UI: ETF Tickers and Teal characteristics.
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
TARGET_R = 3.0
SPRINT_DAYS = 15
NUM_PICKS = 5
NUM_BACKTESTS = 11

ETFS = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB"]
ETF_NAMES = {"XLK":"Technology","XLE":"Energy","XLF":"Financials","XLV":"Healthcare","XLY":"Consumer Discretionary","XLC":"Communication Services","XLP":"Consumer Staples","XLU":"Utilities","XLI":"Industrials","XLRE":"Real Estate","XLB":"Materials"}
SECTOR_TO_ETF = {v: k for k, v in ETF_NAMES.items()}
DB_SECTOR_MAP = {"Technology":"XLK","Energy":"XLE","Financial Services":"XLF","Healthcare":"XLV","Consumer Cyclical":"XLY","Communication Services":"XLC","Consumer Defensive":"XLP","Utilities":"XLU","Industrials":"XLI","Real Estate":"XLRE","Basic Materials":"XLB"}

def calc_ema(values, span):
    return pd.Series(values).ewm(span=span, adjust=False).mean()

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

def calc_rrg_quadrants(all_data, spy_candles, end_idx):
    results = {}
    for pt in ETFS:
        t_candles = all_data.get(pt, {}).get("candles", [])[:end_idx]
        if len(t_candles) < 30: continue
        results[ETF_NAMES[pt]] = get_rrg_quadrant(t_candles, spy_candles)
    return results

def weighted_rs(closes, ref):
    def qr(n):
        try:
            l = min(len(closes), n*63)
            return (closes.iloc[-1]/closes.iloc[-l]) / (ref.iloc[-1]/ref.iloc[-l])
        except: return 0
    return 0.4*qr(1) + 0.2*qr(2) + 0.2*qr(3) + 0.2*qr(4)

def find_top_stocks(all_data, spy_candles, target_etfs, max_candle_idx):
    ref = pd.Series([c["close"] for c in spy_candles])
    candidates = []
    inv_map = {v: k for k, v in DB_SECTOR_MAP.items()}
    target_sectors = {inv_map[etf] for etf in target_etfs if etf in inv_map}
    for ticker, data in all_data.items():
        if ticker=="SPY" or ticker in ETFS or data.get("skip_calc",1)==1: continue
        if data.get("sector","") not in target_sectors: continue
        candles = data.get("candles", [])[:max_candle_idx]
        if len(candles) < 200: continue
        closes = pd.Series([c["close"] for c in candles])
        price = candles[-1]["close"]
        if price < closes.ewm(span=10, adjust=False).mean().iloc[-1]: continue
        rs = weighted_rs(closes, ref)
        candidates.append({"ticker":ticker, "sector":data.get("sector",""), "rs":rs, "price":price})
    candidates.sort(key=lambda x: -x["rs"])
    return candidates[:NUM_PICKS]

def main():
    print("Loading data for Rajesh_Alpha Peak validation...")
    with open(PRICE_DATA, "r") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    
    # Backtest
    min_idx, max_idx = 205, len(spy_all) - SPRINT_DAYS - 2
    test_indices = sorted(random.sample(range(min_idx, max_idx, max(1, (max_idx-min_idx)//12)), NUM_BACKTESTS))
    
    backtests, capital, wins, losses, sat_out = [], PORTFOLIO_START, 0, 0, 0
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        ema21, sma50 = calc_ema([c["close"] for c in spy_slice], 21).iloc[-1], pd.Series([c["close"] for c in spy_slice]).rolling(50).mean().iloc[-1]
        if ema21 < sma50:
            sat_out += 1
            backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"SAT OUT", "pnl":0, "capital_after":capital})
            continue
        rrg_hist = calc_rrg_quadrants(all_data, spy_slice, idx)
        target_etfs = {SECTOR_TO_ETF[s] for s, q in rrg_hist.items() if q in ("Leading", "Improving")}
        picks = find_top_stocks(all_data, spy_slice, target_etfs, idx)
        risk_per = (capital * MAX_RISK_PCT) / len(picks) if picks else 0
        total_pnl, trade_wins, trade_losses = 0, 0, 0
        for p in picks:
            price = p["price"]
            stop_loss, target = round(price * 0.97, 2), round(price * 1.09, 2)
            shares = max(1, int(risk_per / (price * 0.03)))
            candles = all_data[p["ticker"]]["candles"]
            exit_price = price
            for d in range(1, SPRINT_DAYS + 1):
                if idx + d >= len(candles): break
                l, h, c = candles[idx+d]["low"], candles[idx+d]["high"], candles[idx+d]["close"]
                if l <= stop_loss: exit_price = stop_loss; break
                if h >= target: exit_price = target; break
                exit_price = c
            pnl = (exit_price - price) * shares
            total_pnl += pnl
            if pnl > 0: trade_wins += 1; wins += 1
            else: trade_losses += 1; losses += 1
        capital += total_pnl
        backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"TRADED", "pnl":total_pnl, "capital_after":capital, "wins":trade_wins, "losses":trade_losses})

    # Current
    with open(WEB_DATA, "r") as f: web_data = json.load(f)
    closes_now = [c["close"] for c in spy_all]
    ema21_now, sma50_now = calc_ema(closes_now, 21).iloc[-1], pd.Series(closes_now).rolling(50).mean().iloc[-1]
    current_sprint = {"date":web_data.get("last_updated","")[:10], "market":"FAVORABLE" if ema21_now > sma50_now else "UNFAVORABLE", "ema21":round(ema21_now,2), "sma50":round(sma50_now,2), "orders":[]}
    if ema21_now > sma50_now:
        rrg_now = calc_rrg_quadrants(all_data, spy_all, len(spy_all))
        target_etfs = {SECTOR_TO_ETF[s] for s, q in rrg_now.items() if q in ("Leading", "Improving")}
        picks = find_top_stocks(all_data, spy_all, target_etfs, len(spy_all))
        risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
        for p in picks:
            price = p["price"]
            current_sprint["orders"].append({
                "ticker":p["ticker"], "name":p.get("name",""), "sector":f"{p['sector']} ({DB_SECTOR_MAP.get(p['sector'],'??')})",
                "highlights":p.get("highlights",""), "price":price, "buy_stop":price, "buy_limit":round(price*1.002,2),
                "stop":round(price*0.97,2), "stop_limit":round(price*0.968,2), "target":round(price*1.09,2),
                "shares":max(1, int(risk_per/(price*0.03))), "sl_pct":3.0, "target_pct":9.0
            })

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START, "portfolio_current": round(capital,2), "portfolio_target": PORTFOLIO_START*3,
            "win_rate": round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0, "total_return_pct": round((capital-PORTFOLIO_START)/PORTFOLIO_START*100,1),
            "next_sprint_reminder": "Rajesh_Alpha: 5 Picks | 15-Day Cycle | 3R Target."
        },
        "backtests": backtests, "current_sprint": current_sprint
    }
    with open(OUTPUT, "w") as f: json.dump(output, f, indent=2)
    template_path, html_path = os.path.join(ROOT, "sprints_template.html"), os.path.join(ROOT, "output", "sprints.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f: html = f.read()
        html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json.dumps(output).replace("'", "\\'") + "'")
        with open(html_path, "w", encoding="utf-8") as f: f.write(html)
        print("Sprints updated for Peak Performance.")

if __name__ == "__main__": main()
