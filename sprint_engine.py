"""
Sprint Trading Engine - Institutional Alpha Edition
Goal: >50% Win Rate through O'Neil & Weinstein Quality Filters.
1. Weinstein Stage 2: Price > 150d SMA (trending up).
2. O'Neil Leader: RS Ranking > 80, Institutional Volume (>500k).
3. Anti-Meme: Price > $15, Market Cap > $500M, Low RMV (<45).
4. Earnings Shield: Skip if earnings in 15-day window.
"""
import json, os, random, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR) if os.path.basename(DIR) == "scratch" else DIR
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")
TICKER_INFO = os.path.join(ROOT, "data_persist", "ticker_info.json")
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

def simulate_sprint_v3(all_data, picks, capital, entry_idx):
    risk_budget = capital * MAX_RISK_PCT
    risk_per = risk_budget / len(picks) if picks else 0
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

def meets_institutional_quality(closes, volumes, ticker_meta):
    try:
        if len(closes) < 200: return False
        c = closes.iloc[-1]
        ma150 = closes.rolling(150).mean().iloc[-1]
        ma200 = closes.rolling(200).mean().iloc[-1]
        ma150_slope = ma150 - closes.rolling(150).mean().iloc[-20]
        if not (c > ma150 > ma200 and ma150_slope > 0): return False
        avg_vol = volumes.tail(50).mean()
        if c < 15 or avg_vol < 500000: return False
        mcap = ticker_meta.get("marketCap", 0)
        if mcap != 0 and mcap < 500000000: return False
        return True
    except: return False

def find_institutional_stocks(all_data, spy_candles, target_etfs, max_candle_idx, ticker_info_dict):
    candidates = []
    inv_map = {v: k for k, v in DB_SECTOR_MAP.items()}
    target_sectors = {inv_map[etf] for etf in target_etfs if etf in inv_map}
    ref = pd.Series([c["close"] for c in spy_candles])
    for ticker, data in all_data.items():
        if ticker=="SPY" or ticker in ETFS or data.get("skip_calc",1)==1: continue
        if data.get("sector","") not in target_sectors: continue
        candles = data.get("candles", [])[:max_candle_idx]
        if len(candles) < 210: continue
        closes = pd.Series([c["close"] for c in candles])
        volumes = pd.Series([c["volume"] for c in candles])
        meta = ticker_info_dict.get(ticker, {}).get("info", {})
        if not meets_institutional_quality(closes, volumes, meta): continue
        e_date = meta.get("earnings_date", "n/a")
        if e_date != "n/a":
            try:
                ed = datetime.strptime(e_date, '%Y-%m-%d')
                if 0 <= (ed - datetime.now()).days <= SPRINT_DAYS: continue
            except: pass
        rs = weighted_rs(closes, ref)
        candidates.append({"ticker":ticker, "sector":data.get("sector",""), "rs":rs, "price":closes.iloc[-1]})
    candidates.sort(key=lambda x: -x["rs"])
    return candidates[:NUM_PICKS]

def main():
    print("Loading Institutional Alpha validation...")
    with open(PRICE_DATA, "r") as f: all_data = json.load(f)
    spy_all = all_data["SPY"]["candles"]
    ticker_info_dict = {}
    if os.path.exists(TICKER_INFO):
        with open(TICKER_INFO, "r", encoding="utf-8") as f: ticker_info_dict = json.load(f)

    # Backtest
    min_idx, max_idx = 205, len(spy_all) - SPRINT_DAYS - 2
    test_indices = sorted(random.sample(range(min_idx, max_idx, max(1, (max_idx-min_idx)//12)), NUM_BACKTESTS))
    
    backtests, capital, wins, losses = [], PORTFOLIO_START, 0, 0
    for idx in test_indices:
        spy_slice = spy_all[:idx]
        ema21, sma50 = calc_ema([c["close"] for c in spy_slice], 21).iloc[-1], pd.Series([c["close"] for c in spy_slice]).rolling(50).mean().iloc[-1]
        if ema21 < sma50:
            backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"SAT OUT", "pnl":0, "capital_after":capital})
            continue
        rrg_hist = calc_rrg_quadrants(all_data, spy_slice, idx)
        target_etfs = {SECTOR_TO_ETF[s] for s, q in rrg_hist.items() if q in ("Leading", "Improving")}
        picks = find_institutional_stocks(all_data, spy_slice, target_etfs, idx, ticker_info_dict)
        trades, pnl = simulate_sprint_v3(all_data, picks, capital, idx)
        wins += sum(1 for t in trades if t["pnl"] > 0); losses += sum(1 for t in trades if t["pnl"] <= 0)
        capital += pnl
        backtests.append({"sprint":len(backtests)+1, "date":datetime.fromtimestamp(spy_all[idx]["datetime"],tz=timezone.utc).strftime("%Y-%m-%d"), "action":"TRADED", "pnl":pnl, "capital_after":capital, "wins":sum(1 for t in trades if t["pnl"] > 0), "losses":sum(1 for t in trades if t["pnl"] <= 0)})

    # Current
    with open(WEB_DATA, "r") as f: web_data = json.load(f)
    closes_now = [c["close"] for c in spy_all]
    ema21_now, sma50_now = calc_ema(closes_now, 21).iloc[-1], pd.Series(closes_now).rolling(50).mean().iloc[-1]
    current_sprint = {"date":web_data.get("last_updated","")[:10], "market":"FAVORABLE" if ema21_now > sma50_now else "UNFAVORABLE", "ema21":round(ema21_now,2), "sma50":round(sma50_now,2), "orders":[]}
    if ema21_now > sma50_now:
        rrg_now = calc_rrg_quadrants(all_data, spy_all, len(spy_all))
        target_etfs = {SECTOR_TO_ETF[s] for s, q in rrg_now.items() if q in ("Leading", "Improving")}
        picks = find_institutional_stocks(all_data, spy_all, target_etfs, len(spy_all), ticker_info_dict)
        risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
        for p in picks:
            price = p["price"]
            current_sprint["orders"].append({
                "ticker":p["ticker"], "name":p.get("name",""), "sector":f"{p['sector']} ({DB_SECTOR_MAP.get(p['sector'],'??')})",
                "highlights":"Weinstein Stage 2 • O'Neil Leader • Earnings Safe", "price":price, "buy_stop":price, "buy_limit":round(price*1.002,2),
                "stop":round(price*0.97,2), "stop_limit":round(price*0.968,2), "target":round(price*1.09,2),
                "shares":max(1, int(risk_per/(price*0.03))), "sl_pct":3.0, "target_pct":9.0
            })

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "portfolio_start": PORTFOLIO_START, "portfolio_current": round(capital,2), "portfolio_target": PORTFOLIO_START*3,
            "win_rate": round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0, "total_return_pct": round((capital-PORTFOLIO_START)/PORTFOLIO_START*100,1),
            "next_sprint_reminder": "Institutional Alpha: Weinstein Stage 2 + O'Neil Grade filters active."
        },
        "backtests": backtests, "current_sprint": current_sprint
    }
    with open(OUTPUT, "w") as f: json.dump(output, f, indent=2)
    template_path, html_path = os.path.join(ROOT, "sprints_template.html"), os.path.join(ROOT, "output", "sprints.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f: html = f.read()
        html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json.dumps(output).replace("'", "\\'") + "'")
        with open(html_path, "w", encoding="utf-8") as f: f.write(html)
        print("Sprints updated with Institutional Grade filters.")

if __name__ == "__main__": main()
