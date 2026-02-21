# market_regime.py
import pandas as pd
from minervini_engine import sma

def spy_trend_ok(spy_df: pd.DataFrame) -> bool:
    df = spy_df.copy()
    df["ma50"] = sma(df["Close"], 50)
    df["ma200"] = sma(df["Close"], 200)
    last = df.iloc[-1]
    return bool(last["Close"] > last["ma50"] and last["ma50"] > last["ma200"])