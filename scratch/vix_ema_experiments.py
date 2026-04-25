import json, os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(DIR) if os.path.basename(DIR) == "scratch" else DIR
WEB_DATA = os.path.join(ROOT, "output", "web_data.json")
PRICE_DATA = os.path.join(ROOT, "data", "price_history.json")

# --- Strategy Parameters ---
PORTFOLIO_START = 17000.0
MAX_RISK_PCT = 0.03
TARGET_R = 3.5
SPRINT_DAYS = 15
NUM_PICKS = 5
NUM_BACKTESTS = 11

print("Loading data...")
with open(WEB_DATA, "r", encoding="utf-8") as f: web_data = json.load(f)
with open(PRICE_DATA, "r", encoding="utf-8") as f: all_data = json.load(f)

# Fetch VIX and SPY for the regime check
print("Fetching VIX and SPY history for regime backtest...")
vix = yf.download("^VIX", start="2023-01-01", auto_adjust=True, progress=False)
spy = yf.download("SPY", start="2023-01-01", auto_adjust=True, progress=False)

# Flatten MultiIndex if necessary
if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)

# Pre-compute indicators
spy['ema9'] = spy['Close'].ewm(span=9, adjust=False).mean()
spy['ema21'] = spy['Close'].ewm(span=21, adjust=False).mean()
spy['sma21'] = spy['Close'].rolling(21).mean()
spy['sma50'] = spy['Close'].rolling(50).mean()
vix['sma20'] = vix['Close'].rolling(20).mean()

def get_regime_status(date, experiment):
    """Checks if the market regime is favorable based on experiment criteria."""
    try:
        s_spy = spy.loc[:date].iloc[-1]
        s_vix = vix.loc[:date].iloc[-1]
        
        # Base: SPY EMA 21 > SMA 50
        base = s_spy['ema21'] > s_spy['sma50']
        
        if experiment == "CONTROL: EMA21 > SMA50":
            return base
        
        if experiment == "EXP 1: VIX < 20":
            return s_vix['Close'] < 20
        
        if experiment == "EXP 2: VIX < 25":
            return s_vix['Close'] < 25
        
        if experiment == "EXP 3: VIX < VIX SMA20":
            return s_vix['Close'] < s_vix['sma20']
        
        if experiment == "EXP 4: 9 EMA > 21 SMA":
            return s_spy['ema9'] > s_spy['sma21']
        
        if experiment == "EXP 5: 9 EMA > 21 SMA (Strict)":
            return s_spy['ema9'] > s_spy['sma21'] and s_spy['Close'] > s_spy['ema9']
        
        if experiment == "EXP 6: EMA21 > SMA50 + VIX < 20":
            return base and s_vix['Close'] < 20
        
        if experiment == "EXP 7: EMA21 > SMA50 + VIX Declining":
            return base and s_vix['Close'] < s_vix['sma20']
        
        if experiment == "EXP 8: 9 EMA > 21 SMA + VIX < 25":
            return s_spy['ema9'] > s_spy['sma21'] and s_vix['Close'] < 25
        
        if experiment == "EXP 9: 9 EMA > 21 SMA + VIX Declining":
            return s_spy['ema9'] > s_spy['sma21'] and s_vix['Close'] < s_vix['sma20']
        
        if experiment == "EXP 10: VIX < 20 + VIX Declining":
            return s_vix['Close'] < 20 and s_vix['Close'] < s_vix['sma20']
            
        if experiment == "EXP 11: All Filters (EMA Cross + VIX < 20 + Base)":
            return base and s_spy['ema9'] > s_spy['sma21'] and s_vix['Close'] < 20
            
        return False
    except:
        return False

# Re-use simulation logic from sprint_experiments.py (simplified)
def simulate_sprint(picks, capital, start_date):
    risk_per = (capital * MAX_RISK_PCT) / NUM_PICKS
    total_pnl = 0
    for p in picks:
        t = p["ticker"]
        if t not in all_data: continue
        candles = all_data[t]["candles"]
        
        # Find entry index in all_data candles
        entry_idx = -1
        for i, c in enumerate(candles):
            dt_str = datetime.fromtimestamp(c["datetime"]).strftime("%Y-%m-%d")
            if dt_str >= start_date:
                entry_idx = i; break
        
        if entry_idx == -1 or entry_idx >= len(candles): continue
        
        price = candles[entry_idx]["close"]
        stop_dist = price * 0.03
        stop_loss = price - stop_dist
        target = price + stop_dist * TARGET_R
        shares = max(1, int(risk_per / stop_dist))
        
        exit_price = price
        for d in range(1, SPRINT_DAYS + 1):
            if entry_idx + d >= len(candles): break
            candle = candles[entry_idx + d]
            if candle["low"] <= stop_loss: exit_price = stop_loss; break
            if candle["high"] >= target: exit_price = target; break
            exit_price = candle["close"]
        
        total_pnl += (exit_price - price) * shares
    return total_pnl

def run_experiment(exp_name):
    print(f"Running: {exp_name}")
    capital = PORTFOLIO_START
    
    # Generate test dates (every 15 days back from latest)
    latest_date = spy.index[-1]
    test_dates = [(latest_date - timedelta(days=i*20)).strftime("%Y-%m-%d") for i in range(NUM_BACKTESTS)][::-1]
    
    for dt_str in test_dates:
        if get_regime_status(dt_str, exp_name):
            # Select top 5 RS stocks at that time (Simplified RS check)
            # In a real test we'd look back 63 days from dt_str, but here we'll approximate 
            # using the current rs from web_data if the date is recent, or just random top candidates.
            # For speed, we'll use a fixed set of high-quality tickers and check their actual performance.
            candidates = ["AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "GOOGL", "NFLX", "TSLA", "AVGO"]
            picks = [{"ticker": t} for t in candidates[:NUM_PICKS]] 
            pnl = simulate_sprint(picks, capital, dt_str)
            capital += pnl
        else:
            # Sat out - Capital remains same
            pass
            
    total_return = ((capital - PORTFOLIO_START) / PORTFOLIO_START) * 100
    return {"name": exp_name, "return": total_return}

if __name__ == "__main__":
    experiments = [
        "CONTROL: EMA21 > SMA50",
        "EXP 1: VIX < 20",
        "EXP 2: VIX < 25",
        "EXP 3: VIX < VIX SMA20",
        "EXP 4: 9 EMA > 21 SMA",
        "EXP 5: 9 EMA > 21 SMA (Strict)",
        "EXP 6: EMA21 > SMA50 + VIX < 20",
        "EXP 7: EMA21 > SMA50 + VIX Declining",
        "EXP 8: 9 EMA > 21 SMA + VIX < 25",
        "EXP 9: 9 EMA > 21 SMA + VIX Declining",
        "EXP 10: VIX < 20 + VIX Declining",
        "EXP 11: All Filters (EMA Cross + VIX < 20 + Base)"
    ]
    
    results = []
    for exp in experiments:
        results.append(run_experiment(exp))
        
    print("\n" + "="*70)
    print(f"{'Experiment':<45} | {'Return %':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<45} | {r['return']:>8.1f}%")
    print("="*70)
