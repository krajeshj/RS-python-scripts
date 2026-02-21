# minervini_engine.py
import pandas as pd
import numpy as np

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ema(df["Close"], 20)
    df["ma50"]  = sma(df["Close"], 50)
    df["ma150"] = sma(df["Close"], 150)
    df["ma200"] = sma(df["Close"], 200)
    df["ma200_slope"] = df["ma200"] - df["ma200"].shift(20)

    df["atr14"] = atr(df, 14)
    df["atr14_pct"] = df["atr14"] / df["Close"]

    df["vol20"] = df["Volume"].rolling(20).mean()

    df["range5"]  = df["High"].rolling(5).max() - df["Low"].rolling(5).min()
    df["range10"] = df["High"].rolling(10).max() - df["Low"].rolling(10).min()
    df["atr14_20d_avg"] = df["atr14"].rolling(20).mean()

    df["contraction"] = (df["atr14"] < df["atr14_20d_avg"]) & (df["range5"] < df["range10"])

    df["pivot_high_5d"] = df["High"].rolling(5).max().shift(1)
    df["confirmed_breakout"] = (df["Close"] > df["pivot_high_5d"]) & (df["Volume"] > df["vol20"] * 1.3)

    df["not_extended"] = (df["Close"] / df["ma50"]) < 1.08
    return df

def compute_weekly_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    # df_daily index must be datetime
    w = df_daily.resample("W-FRI").agg({"Close":"last"})
    w["weekly_ma10"] = w["Close"].rolling(10).mean()
    w = w.rename(columns={"Close":"weekly_close"})
    return w

def compute_ptc(latest_daily: pd.Series, latest_weekly: pd.Series) -> int:
    flags = [
        latest_daily["Close"] > latest_daily["ema20"],
        latest_daily["Close"] > latest_daily["ma50"],
        latest_daily["ema20"]  > latest_daily["ma50"],
        latest_daily["ma50"]   > latest_daily["ma150"],
        latest_daily["ma150"]  > latest_daily["ma200"],
        latest_daily["ma200_slope"] > 0,
        latest_weekly["weekly_close"] > latest_weekly["weekly_ma10"],
    ]
    return int(sum(bool(x) for x in flags))

def is_high_win_candidate(
    latest_daily: pd.Series,
    ptc: int,
    rs_percentile: float,
    rmv: float,
    spy_trend_ok: bool,
    days_to_earnings: int | None = None,
) -> bool:
    trend_ok = (
        latest_daily["Close"] > latest_daily["ma50"] and
        latest_daily["ma50"]  > latest_daily["ma150"] and
        latest_daily["ma150"] > latest_daily["ma200"] and
        latest_daily["ma200_slope"] > 0
    )
    rs_ok = rs_percentile >= 97
    rmv_ok = rmv < 50
    ptc_ok = ptc >= 6
    earnings_ok = (days_to_earnings is None) or (days_to_earnings > 10)

    return bool(
        spy_trend_ok and
        trend_ok and rs_ok and rmv_ok and ptc_ok and earnings_ok and
        bool(latest_daily["contraction"]) and
        bool(latest_daily["confirmed_breakout"]) and
        bool(latest_daily["not_extended"])
    )
    