import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Paths
DIR = os.path.dirname(os.path.realpath(__file__))
PRICE_DATA = os.path.join(DIR, "data", "price_history.json")
HISTORY_FILE = os.path.join(DIR, "output", "stock_history.json")
REFERENCE_TICKER = "SPY"
HISTORY_WINDOW = 10 

def quarters_rs(closes: pd.Series, closes_ref: pd.Series, n):
    try:
        length = min(len(closes), n * int(252 / 4))
        df_prices_n = closes.tail(length).dropna()
        if df_prices_n.empty: return 0
        prices_n = df_prices_n.head(1).item()

        df_prices_ref_n = closes_ref.tail(length).dropna()
        if df_prices_ref_n.empty: return 0
        prices_ref_n = df_prices_ref_n.head(1).item()

        prices = closes.tail(1).item()
        prices_ref = closes_ref.tail(1).item()

        rs_n = (prices / prices_n) / (prices_ref / prices_ref_n)
        return rs_n
    except:
        return 0

def relative_strength(closes: pd.Series, closes_ref: pd.Series):
    try:
        rs1 = quarters_rs(closes, closes_ref, 1)
        rs2 = quarters_rs(closes, closes_ref, 2)
        rs3 = quarters_rs(closes, closes_ref, 3)
        rs4 = quarters_rs(closes, closes_ref, 4)
        rs = 0.4 * rs1 + 0.2 * rs2 + 0.2 * rs3 + 0.2 * rs4
        return rs
    except:
        return 1e-14

def process_day(day_offset, all_data, tickers):
    """Calculates RS Raw for all tickers at a specific day offset."""
    day_ref_candles = all_data[REFERENCE_TICKER]["candles"]
    if day_offset > 0:
        day_ref_candles = day_ref_candles[:-day_offset]
    
    if not day_ref_candles:
        return None
        
    ref_closes = pd.Series([c["close"] for c in day_ref_candles])
    
    day_scores = {}
    for t in tickers:
        try:
            candles = all_data[t]["candles"]
            if day_offset > 0:
                candles = candles[:-day_offset]
            
            if not candles or len(candles) < 120:
                continue
                
            closes = pd.Series([c["close"] for c in candles])
            score = relative_strength(closes, ref_closes)
            day_scores[t] = score
        except:
            continue
        
    symbols = list(day_scores.keys())
    scores = [day_scores[s] for s in symbols]
    
    if not scores:
        return {}
        
    df = pd.DataFrame({"symbol": symbols, "score": scores})
    df["rank"] = pd.qcut(df["score"], 100, labels=False, duplicates='drop', precision=64) + 1
    
    return dict(zip(df["symbol"], df["rank"].astype(int)))

def main():
    print(f"Loading {PRICE_DATA}...")
    if not os.path.exists(PRICE_DATA):
        print("Price data not found!")
        return
        
    with open(PRICE_DATA, "r") as f:
        all_data = json.load(f)
    
    tickers = [t for t in all_data.keys() if t != REFERENCE_TICKER and all_data[t].get("skip_calc", 1) == 0]
    
    print(f"Generating RS history for {len(tickers)} stocks over {HISTORY_WINDOW} days...")
    
    history_data = {t: [] for t in tickers}
    
    for i in range(HISTORY_WINDOW):
        print(f"  Processing day T-{i}...")
        ranks = process_day(i, all_data, tickers)
        if ranks:
            for t in tickers:
                if t in ranks:
                    history_data[t].append(ranks[t])
                else:
                    if history_data[t]: # Use previous day's rank if available
                        history_data[t].append(history_data[t][-1])
                    else:
                        history_data[t].append(50)
        else:
            # If processing failed for this day, pad with previous
            for t in tickers:
                if history_data[t]:
                    history_data[t].append(history_data[t][-1])
                else:
                    history_data[t].append(50)
                    
    print(f"Saving history to {HISTORY_FILE}...")
    with open(HISTORY_FILE, "w") as f:
        json.dump(history_data, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
