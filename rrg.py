import json, os
import pandas as pd
DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(DIR, "data", "price_history.json")) as f:
    json_data = json.load(f)

rrg_history = {}
etfs = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC", "XLP", "XLU", "XLI", "XLRE", "XLB", "SMH", "XBI", "XHB", "XRT", "XME"]

spy_candles = json_data.get("SPY", {}).get("candles", [])
if spy_candles:
    # Use pandas Series
    spy_closes = pd.Series([c["close"] for c in spy_candles])
    
    for pt in etfs:
        t_candles = json_data.get(pt, {}).get("candles", [])
        if len(t_candles) > 30 and len(t_candles) <= len(spy_closes):
            try:
                # Align lengths just in case of tiny discrepancies
                min_len = min(len(t_candles), len(spy_closes))
                tc = pd.Series([c["close"] for c in t_candles[-min_len:]])
                sc = spy_closes.tail(min_len).reset_index(drop=True)

                # RRG Math Approximation
                rs_raw = tc / sc
                
                # JdK RS-Ratio
                sma_ratio = rs_raw.rolling(14).mean()
                # Normalize via z-score and offset it back to 100 center
                rs_ratio = 100 + ((sma_ratio - sma_ratio.rolling(14).mean()) / sma_ratio.rolling(14).std().replace(0, 1)) * 5
                
                # JdK RS-Momentum
                # ROC of RS-Ratio
                roc = rs_ratio.diff(10)
                # Normalize ROC
                rs_mom = 100 + ((roc - roc.rolling(10).mean()) / roc.rolling(10).std().replace(0, 1)) * 5
                
                # Extract trailing 15 points
                r_r = rs_ratio.tail(15).fillna(100).tolist()
                r_m = rs_mom.tail(15).fillna(100).tolist()
                
                # RRG is crosshair centered exactly at 0,0 locally
                rrg_history[pt] = [{"x": round(r_r[i] - 100, 2), "y": round(r_m[i] - 100, 2)} for i in range(len(r_r))]
            except Exception as e:
                print(f"Failed {pt}: {e}")

with open(os.path.join(DIR, "output", "rrg.json"), "w") as f:
    json.dump({"date": "latest", "data": rrg_history}, f)
print("Saved RRG history")
