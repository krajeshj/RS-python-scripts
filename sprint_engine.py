"""
Sprint Trading Engine
Generates backtest results + current sprint recommendations.
Outputs sprint_data.json for the dashboard.
Market Filter: SPY 21 EMA > 50 SMA = trade, else sit out.
Strategy: 5R OCO on top RS stocks from Leading/Improving RRG sectors.
"""
import json, os, random, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR) if os.path.basename(DIR) == "scratch" else DIR
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")
OUTPUT = os.path.join(ROOT, "output", "sprint_data.json")

# --- Strategy Parameters ---
PORTFOLIO_START = 17000.0
MAX_RISK_PCT = 0.03
TARGET_R = 3
SPRINT_DAYS = 15       # trading days per sprint
NUM_PICKS = 5
NUM_BACKTESTS = 11
MAX_STOP_PCT = 0.04    # 4% max stop

# --- Sector ETFs ---
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

# ============================================================================
# Core Math
# ============================================================================
def calc_ema(values, span):
    s = pd.Series(values)
    return s.ewm(span=span, adjust=False).mean()

def calc_sma(values, window):
    s = pd.Series(values)
    return s.rolling(window).mean()

def market_is_favorable(spy_candles):
    """21 EMA > 50 SMA on SPY = favorable."""
    closes = [c["close"] for c in spy_candles]
    if len(closes) < 50:
        return False, 0, 0
    ema21 = calc_ema(closes, 21).iloc[-1]
    sma50 = calc_sma(closes, 50).iloc[-1]
    return ema21 > sma50, round(ema21, 2), round(sma50, 2)

def calc_rrg_quadrants(all_data, spy_candles):
    """Returns dict of {etf: quadrant} for sector ETFs."""
    spy_closes = pd.Series([c["close"] for c in spy_candles])
    results = {}
    for pt in ETFS:
        t_candles = all_data.get(pt, {}).get("candles", [])
        if len(t_candles) <= 30 or len(t_candles) > len(spy_closes):
            continue
        try:
            min_len = min(len(t_candles), len(spy_closes))
            tc = pd.Series([c["close"] for c in t_candles[-min_len:]])
            sc = spy_closes.tail(min_len).reset_index(drop=True)
            rs_raw = tc / sc
            sma_r = rs_raw.rolling(14).mean()
            rs_ratio = 100 + ((sma_r - sma_r.rolling(14).mean()) / sma_r.rolling(14).std().replace(0, 1)) * 5
            roc = rs_ratio.diff(10)
            rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
            x = round(rs_ratio.iloc[-1] - 100, 2)
            y = round(rs_mom.iloc[-1] - 100, 2)
            if x > 0 and y > 0:
                q = "Leading"
            elif x < 0 and y > 0:
                q = "Improving"
            elif x > 0 and y < 0:
                q = "Weakening"
            else:
                q = "Lagging"
            results[pt] = {"x": x, "y": y, "quadrant": q, "sector": ETF_NAMES[pt]}
        except:
            pass
    return results

def quarters_rs(closes, ref, n):
    try:
        length = min(len(closes), n * 63)
        pn = closes.tail(length).head(1).item()
        rn = ref.tail(length).head(1).item()
        return (closes.iloc[-1] / pn) / (ref.iloc[-1] / rn)
    except:
        return 0

def relative_strength(closes, ref):
    return 0.4*quarters_rs(closes,ref,1) + 0.2*quarters_rs(closes,ref,2) + 0.2*quarters_rs(closes,ref,3) + 0.2*quarters_rs(closes,ref,4)

def calc_atr(candles, n=14):
    if len(candles) < n + 1:
        return None
    trs = []
    for i in range(-n, 0):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return np.mean(trs)

def find_top_stocks(all_data, spy_candles, rrg, max_candle_idx):
    """Find top RS stocks from Leading + Improving sectors."""
    target_sectors = set()
    for etf, info in rrg.items():
        if info["quadrant"] in ("Leading", "Improving"):
            sector_name = ETF_NAMES.get(etf, "")
            if sector_name in SECTOR_MAP:
                target_sectors.add(SECTOR_MAP[sector_name])

    if not target_sectors:
        return []

    ref = pd.Series([c["close"] for c in spy_candles])
    candidates = []

    for ticker, data in all_data.items():
        if ticker == "SPY" or data.get("skip_calc", 1) == 1:
            continue
        if data.get("sector", "") not in target_sectors:
            continue
        candles = data.get("candles", [])[:max_candle_idx]
        if len(candles) < 200:
            continue

        closes = pd.Series([c["close"] for c in candles])
        rs = relative_strength(closes, ref)
        price = candles[-1]["close"]
        atr = calc_atr(candles)
        if atr is None:
            continue

        sma50 = np.mean([c["close"] for c in candles[-50:]])
        sma200 = np.mean([c["close"] for c in candles[-200:]])
        ema8 = closes.ewm(span=8, adjust=False).mean().iloc[-1]
        ext_50 = ((price - sma50) / sma50) * 100
        atr_pct = (atr / price) * 100
        high_52w = max(c["high"] for c in candles[-252:])
        pct_from_high = ((price - high_52w) / high_52w) * 100

        if price < sma50 or price < sma200:
            continue
        if ext_50 > 12:
            continue
        if pct_from_high < -15:
            continue
        if price < ema8:
            continue
        if atr_pct > 5:
            continue

        candidates.append({
            "ticker": ticker,
            "name": data.get("name", ticker)[:25],
            "sector": data.get("sector", ""),
            "rs": rs,
            "price": round(price, 2),
            "atr": round(atr, 2),
            "atr_pct": round(atr_pct, 1),
            "sma50": round(sma50, 2),
            "ema8": round(ema8, 2),
            "ext_50": round(ext_50, 1),
            "pct_high": round(pct_from_high, 1),
        })

    candidates.sort(key=lambda x: -x["rs"])
    return candidates[:NUM_PICKS]

def simulate_sprint(all_data, picks, capital, entry_candle_idx):
    """Simulate OCO orders over SPRINT_DAYS trading days."""
    risk_budget = capital * MAX_RISK_PCT
    risk_per_pick = risk_budget / len(picks) if picks else 0
    trades = []
    total_pnl = 0

    for p in picks:
        ticker = p["ticker"]
        candles = all_data[ticker]["candles"]
        price = p["price"]
        atr = p["atr"]

        stop_dist = min(atr, price * MAX_STOP_PCT)
        stop_loss = round(price - stop_dist, 2)
        target = round(price + stop_dist * TARGET_R, 2)
        sl_pct = round((stop_dist / price) * 100, 1)

        shares = max(1, int(risk_per_pick / stop_dist))
        cost = shares * price

        exit_price = price
        exit_reason = "OPEN"
        days_held = SPRINT_DAYS

        for d in range(1, SPRINT_DAYS + 1):
            idx = entry_candle_idx + d
            if idx >= len(candles):
                exit_price = candles[-1]["close"]
                exit_reason = "END OF DATA"
                days_held = d
                break
            day_low = candles[idx]["low"]
            day_high = candles[idx]["high"]
            if day_low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP"
                days_held = d
                break
            if day_high >= target:
                exit_price = target
                exit_reason = "TARGET"
                days_held = d
                break
            exit_price = candles[idx]["close"]

        if exit_reason == "OPEN":
            exit_price = candles[min(entry_candle_idx + SPRINT_DAYS, len(candles)-1)]["close"]
            exit_reason = "EXPIRED"

        pnl = (exit_price - price) * shares
        pnl_pct = round(((exit_price - price) / price) * 100, 1)
        total_pnl += pnl

        trades.append({
            "ticker": ticker,
            "name": p["name"],
            "sector": p["sector"],
            "shares": shares,
            "entry": price,
            "stop": stop_loss,
            "sl_pct": sl_pct,
            "target": target,
            "target_pct": round((stop_dist * TARGET_R / price) * 100, 1),
            "exit": round(exit_price, 2),
            "result": exit_reason,
            "days": days_held,
            "pnl": round(pnl, 2),
            "pnl_pct": pnl_pct,
            "cost": round(cost, 2),
            "risk": round(shares * stop_dist, 2),
        })

    return trades, round(total_pnl, 2)

# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading price data...")
    with open(PRICE_DATA, "r") as f:
        all_data = json.load(f)

    spy_all = all_data["SPY"]["candles"]
    total_days = len(spy_all)
    print(f"SPY has {total_days} trading days of history")

    # Need at least 200 days warmup + SPRINT_DAYS buffer from end
    min_idx = 205
    max_idx = total_days - SPRINT_DAYS - 2
    available = max_idx - min_idx

    # Space 11 tests evenly across whatever history we have
    if available < NUM_BACKTESTS:
        print(f"WARNING: Only {available} viable days. Running {available} tests.")
        test_indices = list(range(min_idx, max_idx))
    else:
        step = max(1, available // (NUM_BACKTESTS + 1))
        random.seed(42)
        candidates = list(range(min_idx, max_idx, max(step, 5)))
        random.shuffle(candidates)
        test_indices = sorted(candidates[:NUM_BACKTESTS])

    print(f"\nRunning {NUM_BACKTESTS} backtests at indices: {test_indices}")

    backtests = []
    capital = PORTFOLIO_START
    wins = 0
    losses = 0
    sat_out = 0

    for test_num, entry_idx in enumerate(test_indices):
        spy_slice = spy_all[:entry_idx]
        entry_date = spy_all[entry_idx].get("datetime", 0)
        if entry_date:
            date_str = datetime.fromtimestamp(entry_date, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            date_str = f"Day-{entry_idx}"

        # Market filter
        favorable, ema21, sma50 = market_is_favorable(spy_slice)
        spy_price = spy_slice[-1]["close"]

        if not favorable:
            sat_out += 1
            backtests.append({
                "sprint": test_num + 1,
                "date": date_str,
                "spy_price": round(spy_price, 2),
                "ema21": ema21,
                "sma50": sma50,
                "market": "UNFAVORABLE",
                "action": "SAT OUT",
                "sectors": [],
                "trades": [],
                "pnl": 0,
                "capital_before": round(capital, 2),
                "capital_after": round(capital, 2),
            })
            print(f"  Sprint {test_num+1} ({date_str}): SAT OUT - 21EMA ${ema21} < 50SMA ${sma50}")
            continue

        # RRG at this point in time
        # Build trimmed data for RRG
        trimmed = {}
        for t, d in all_data.items():
            c = d.get("candles", [])
            trimmed[t] = {**d, "candles": c[:entry_idx]}

        rrg = calc_rrg_quadrants(trimmed, spy_slice)
        active_sectors = [{"etf": k, **v} for k, v in rrg.items() if v["quadrant"] in ("Leading", "Improving")]

        picks = find_top_stocks(all_data, spy_slice, rrg, entry_idx)

        if not picks:
            sat_out += 1
            backtests.append({
                "sprint": test_num + 1,
                "date": date_str,
                "spy_price": round(spy_price, 2),
                "ema21": ema21,
                "sma50": sma50,
                "market": "FAVORABLE",
                "action": "NO PICKS",
                "sectors": [s["etf"] for s in active_sectors],
                "trades": [],
                "pnl": 0,
                "capital_before": round(capital, 2),
                "capital_after": round(capital, 2),
            })
            print(f"  Sprint {test_num+1} ({date_str}): No qualifying picks found")
            continue

        trades, pnl = simulate_sprint(all_data, picks, capital, entry_idx)

        sprint_wins = sum(1 for t in trades if t["pnl"] > 0)
        sprint_losses = sum(1 for t in trades if t["pnl"] <= 0)
        wins += sprint_wins
        losses += sprint_losses

        capital_before = capital
        capital += pnl

        backtests.append({
            "sprint": test_num + 1,
            "date": date_str,
            "spy_price": round(spy_price, 2),
            "ema21": ema21,
            "sma50": sma50,
            "market": "FAVORABLE",
            "action": "TRADED",
            "sectors": [s["etf"] for s in active_sectors],
            "trades": trades,
            "pnl": pnl,
            "pnl_pct": round((pnl / capital_before) * 100, 2),
            "capital_before": round(capital_before, 2),
            "capital_after": round(capital, 2),
            "wins": sprint_wins,
            "losses": sprint_losses,
        })
        print(f"  Sprint {test_num+1} ({date_str}): P&L ${pnl:+,.2f} | W:{sprint_wins} L:{sprint_losses} | Capital: ${capital:,.2f}")

    # --- Current sprint (TODAY) ---
    print("\n--- Generating CURRENT sprint recommendations ---")
    favorable, ema21_now, sma50_now = market_is_favorable(spy_all)
    spy_now = spy_all[-1]["close"]
    spy_date = datetime.fromtimestamp(spy_all[-1].get("datetime", 0), tz=timezone.utc).strftime("%Y-%m-%d")

    current_sprint = {
        "date": spy_date,
        "spy_price": round(spy_now, 2),
        "ema21": ema21_now,
        "sma50": sma50_now,
        "market": "FAVORABLE" if favorable else "UNFAVORABLE",
    }

    if favorable:
        rrg_now = calc_rrg_quadrants(all_data, spy_all)
        active_now = [{"etf": k, **v} for k, v in rrg_now.items() if v["quadrant"] in ("Leading", "Improving")]
        picks_now = find_top_stocks(all_data, spy_all, rrg_now, len(spy_all))
        
        current_sprint["sectors"] = active_now
        current_sprint["rrg_all"] = [{"etf": k, **v} for k, v in rrg_now.items()]
        
        if picks_now:
            risk_budget = capital * MAX_RISK_PCT
            risk_per = risk_budget / len(picks_now)
            orders = []
            for p in picks_now:
                stop_dist = min(p["atr"], p["price"] * MAX_STOP_PCT)
                stop_loss = round(p["price"] - stop_dist, 2)
                target = round(p["price"] + stop_dist * TARGET_R, 2)
                shares = max(1, int(risk_per / stop_dist))
                orders.append({
                    **p,
                    "shares": shares,
                    "stop": stop_loss,
                    "sl_pct": round((stop_dist / p["price"]) * 100, 1),
                    "target": target,
                    "target_pct": round((stop_dist * TARGET_R / p["price"]) * 100, 1),
                    "risk": round(shares * stop_dist, 2),
                    "potential": round(shares * stop_dist * TARGET_R, 2),
                    "cost": round(shares * p["price"], 2),
                })
            current_sprint["orders"] = orders
            current_sprint["action"] = "ENTER NEW SPRINT"
        else:
            current_sprint["orders"] = []
            current_sprint["action"] = "NO QUALIFYING PICKS"
    else:
        current_sprint["sectors"] = []
        current_sprint["orders"] = []
        current_sprint["action"] = "SIT OUT - Market unfavorable (21 EMA < 50 SMA)"

    # --- Summary ---
    traded_sprints = [b for b in backtests if b["action"] == "TRADED"]
    total_pnl = sum(b["pnl"] for b in traded_sprints)

    summary = {
        "portfolio_start": PORTFOLIO_START,
        "portfolio_current": round(capital, 2),
        "portfolio_target": PORTFOLIO_START * 3,
        "target_r": TARGET_R,
        "max_risk_pct": MAX_RISK_PCT,
        "sprint_days": SPRINT_DAYS,
        "total_backtests": NUM_BACKTESTS,
        "traded": len(traded_sprints),
        "sat_out": sat_out,
        "total_wins": wins,
        "total_losses": losses,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((capital - PORTFOLIO_START) / PORTFOLIO_START * 100, 1),
        "next_sprint_reminder": "Cycle every 10 trading days. Next sprint review in 10 trading days.",
    }

    output = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": summary,
        "backtests": backtests,
        "current_sprint": current_sprint,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT}")

    # Inject data into sprints.html from template
    template_path = os.path.join(ROOT, "sprints_template.html")
    html_path = os.path.join(ROOT, "output", "sprints.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
        json_str = json.dumps(output).replace("'", "\\'")
        html = html.replace("'__SPRINT_DATA_PLACEHOLDER__'", "'" + json_str + "'")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Injected data into {html_path}")
    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"Sprints Traded: {len(traded_sprints)} | Sat Out: {sat_out}")
    print(f"Wins: {wins} | Losses: {losses} | Win Rate: {summary['win_rate']}%")
    print(f"Total P&L: ${total_pnl:+,.2f} ({summary['total_return_pct']:+.1f}%)")
    print(f"Capital: ${PORTFOLIO_START:,.2f} -> ${capital:,.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
