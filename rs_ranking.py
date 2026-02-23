import sys
import argparse
import pandas as pd
import numpy as np
import json
import os
import csv
import yaml
from datetime import datetime
from functools import reduce

from rs_data import cfg, read_json as read_json_from_rs_data
from concurrent.futures import ProcessPoolExecutor, as_completed

DIR = os.path.dirname(os.path.realpath(__file__))

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

try:
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
except FileNotFoundError:
    config = None
except yaml.YAMLError as exc:
    print(exc)

PRICE_DATA = os.path.join(DIR, "data", "price_history.json")
MIN_PERCENTILE = cfg("MIN_PERCENTILE")
POS_COUNT_TARGET = cfg("POSITIONS_COUNT_TARGET")
REFERENCE_TICKER = cfg("REFERENCE_TICKER")
ALL_STOCKS = cfg("USE_ALL_LISTED_STOCKS")
TICKER_INFO_FILE = os.path.join(DIR, "data_persist", "ticker_info.json")
TICKER_INFO_DICT = read_json_from_rs_data(TICKER_INFO_FILE)

TITLE_RANK = "Rank"
TITLE_TICKER = "Ticker"
TITLE_TICKERS = "Tickers"
TITLE_MINERVINI = "Minervini"
TITLE_SECTOR = "Sector"
TITLE_INDUSTRY = "Industry"
TITLE_UNIVERSE = "Universe" if not ALL_STOCKS else "Exchange"
TITLE_PERCENTILE = "Percentile"
TITLE_1M = "1 Month Ago"
TITLE_3M = "3 Months Ago"
TITLE_6M = "6 Months Ago"
TITLE_RS = "Relative Strength"
TITLE_RMV = "RMV"

# --- New signal columns ---
TITLE_CLOSE = "Close"
TITLE_ATR_PCT = "ATR% (14)"
TITLE_PTC = "PTC"
TITLE_CONTRACTION = "Contraction"
TITLE_BREAKOUT = "Confirmed Breakout"
TITLE_NOT_EXT = "Not Extended"
TITLE_FLIP = "Flip Setup"
TITLE_SPY_OK = "SPY Trend OK"
TITLE_CAND_HIGH = "Candidate High-Win"
TITLE_CAND_FLIP = "Candidate Flip"
TITLE_SOURCE = "Source"
TITLE_CANSLIM = "CANSLIM"
TITLE_DTE = "DaysToEarnings"
TITLE_NAME = "Name"
TITLE_COMMENTARY = "Commentary"
TITLE_STAGE = "Stage"

# RMV thresholds calibrated to your RMV formula (volatility_pct*7)
CONTROLLED_RMV_MAX = 45   # avg daily range <= ~6.4%
ULTRA_RMV_MAX = 35        # avg daily range <= ~5.0%
FLIP_RMV_MAX = 40         # stricter for flip

# "Controlled setups only" guardrails (toggle as you like)
EXCLUDE_INDUSTRIES = set([
    "Biotechnology",
    "Other Precious Metals & Mining",
    "Silver",
    "Gold",
    "Uranium",
    "Oil & Gas Drilling",
    "Oil & Gas Equipment & Services",
])

if not os.path.exists('output'):
    os.makedirs('output')


def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as fp:
        return json.load(fp)


def calculate_rmv(closes: pd.Series, highs: pd.Series = None, lows: pd.Series = None, lookback_period: int = 15):
    """
    RMV = scaled avg daily range % over lookback.
    volatility_pct = mean(high-low)/mean(close)*100
    rmv = clamp(volatility_pct*7, 0..100)
    """
    try:
        if len(closes) < lookback_period:
            return 50.0

        if highs is None:
            highs = closes
        if lows is None:
            lows = closes

        min_length = min(len(closes), len(highs), len(lows))
        closes = closes.tail(min_length)
        highs = highs.tail(min_length)
        lows = lows.tail(min_length)

        if len(closes) < lookback_period:
            return 50.0

        daily_ranges = highs - lows
        avg_price = closes.tail(lookback_period).mean()
        avg_daily_range = daily_ranges.tail(lookback_period).mean()

        if avg_price == 0 or pd.isna(avg_price) or pd.isna(avg_daily_range):
            return 50.0

        volatility_pct = (avg_daily_range / avg_price) * 100
        rmv = min(100.0, max(0.0, volatility_pct * 7.0))

        if pd.isna(rmv) or np.isinf(rmv):
            return 50.0

        return round(float(rmv), 2)
    except Exception as e:
        print(f"Error calculating RMV: {e}")
        return 50.0


def quarters_rs(closes: pd.Series, closes_ref: pd.Series, n):
    try:
        length = min(len(closes), n * int(252 / 4))
        df_prices_n = closes.tail(length).dropna()
        prices_n = df_prices_n.head(1).item()

        df_prices_ref_n = closes_ref.tail(length).dropna()
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
        print("Exception in relative strength - rs = 0")
        return 1e-14


# -------------------------
# Signal helpers (daily/weekly features)
# -------------------------
def _to_df_from_candles(candles):
    if not candles:
        return None
    df = pd.DataFrame(candles)
    if "datetime" in df.columns:
        # rs_data uses epoch seconds
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df = df.sort_values("datetime").set_index("datetime")
    else:
        # fallback
        df = df.reset_index(drop=True)

    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }, inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            return None

    return df


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()


def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _compute_daily_features(df):
    df = df.copy()
    df["ema20"] = _ema(df["Close"], 20)
    df["ma50"] = _sma(df["Close"], 50)
    df["ma150"] = _sma(df["Close"], 150)
    df["ma200"] = _sma(df["Close"], 200)
    df["ma200_slope"] = df["ma200"] - df["ma200"].shift(20)

    df["atr14"] = _atr(df, 14)
    df["atr14_pct"] = df["atr14"] / df["Close"]

    df["vol20"] = df["Volume"].rolling(20).mean()

    df["range5"] = df["High"].rolling(5).max() - df["Low"].rolling(5).min()
    df["range10"] = df["High"].rolling(10).max() - df["Low"].rolling(10).min()
    df["atr14_20d_avg"] = df["atr14"].rolling(20).mean()

    df["contraction"] = (df["atr14"] < df["atr14_20d_avg"]) & (df["range5"] < df["range10"])

    df["pivot_high_5d"] = df["High"].rolling(5).max().shift(1)
    df["confirmed_breakout"] = (df["Close"] > df["pivot_high_5d"]) & (df["Volume"] > df["vol20"] * 1.3)

    df["not_extended"] = (df["Close"] / df["ma50"]) < 1.08
    return df


def _compute_weekly(df_daily):
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        return None
    w = df_daily.resample("W-FRI").agg({"Close": "last"})
    w["weekly_ma10"] = w["Close"].rolling(10).mean()
    w.rename(columns={"Close": "weekly_close"}, inplace=True)
    return w


def _compute_ptc(latest_daily, latest_weekly):
    flags = [
        latest_daily["Close"] > latest_daily["ema20"],
        latest_daily["Close"] > latest_daily["ma50"],
        latest_daily["ema20"] > latest_daily["ma50"],
        latest_daily["ma50"] > latest_daily["ma150"],
        latest_daily["ma150"] > latest_daily["ma200"],
        latest_daily["ma200_slope"] > 0,
        latest_weekly["weekly_close"] > latest_weekly["weekly_ma10"],
    ]
    return int(sum(bool(x) for x in flags))


def _spy_trend_ok(ref_df_daily):
    d = ref_df_daily.copy()
    d["ma50"] = _sma(d["Close"], 50)
    d["ma200"] = _sma(d["Close"], 200)
    last = d.iloc[-1]
    return bool(last["Close"] > last["ma50"] and last["ma50"] > last["ma200"])


def _get_trend_commentary(tdf):
    """Generates one-line commentary based on technicals."""
    try:
        last = tdf.iloc[-1]
        c = last["Close"]
        ma50 = last["ma50"]
        ma200 = last["ma200"]
        
        # Trend indicators
        above_ma50 = c > ma50
        above_ma200 = c > ma200
        golden_cross = ma50 > ma200
        ma200_slope = ma200 - tdf["ma200"].iloc[-20] if len(tdf) > 20 else 0
        
        status = "Sideways"
        comment = "Consolidating near Moving Averages."
        
        if above_ma50 and above_ma200 and golden_cross:
            status = "Upward"
            if ma200_slope > 0:
                comment = "Strong bull trend, trending above MA50 & MA200."
            else:
                comment = "Bullish posture but MA200 flattening."
        elif not above_ma50 and not above_ma200:
            status = "Downward"
            comment = "Bearish posture, trading below major MAs."
        elif above_ma200 and not above_ma50:
            status = "Sideways"
            comment = "Weakening, slipped below MA50 but holding MA200."
        elif above_ma50 and not above_ma200:
            status = "Sideways"
            comment = "Recovering, trading above MA50 but below MA200."
            
        return status, comment
    except:
        return "Unknown", "Insufficient data for trend analysis."


def _get_standard_stage(tdf):
    """
    Stan Weinstein’s 4-Stage Analysis (Heuristic based on MA200)
    1: Basing, 2: Advancing, 3: Topping, 4: Declining
    """
    try:
        last = tdf.iloc[-1]
        c = last["Close"]
        ma200 = last["ma200"]
        ma200_prev = tdf["ma200"].iloc[-20] if len(tdf) > 20 else ma200
        slope = ma200 - ma200_prev

        if c > ma200:
            return 2 if slope > 0 else 3
        else:
            return 4 if slope < 0 else 1
    except:
        return 1



# -------------------------
# Web Export Helpers
# -------------------------
def _get_highlights(row):
    highlights = []
    if row.get(TITLE_BREAKOUT): highlights.append("Confirmed Breakout")
    if row.get(TITLE_CONTRACTION): highlights.append("VCP Contraction")
    if row.get(TITLE_FLIP): highlights.append("Flip Setup")
    if row.get(TITLE_NOT_EXT): highlights.append("Not Extended")
    if row.get(TITLE_PTC, 0) >= 7: highlights.append("Strong Trend Alignment")
    if row.get(TITLE_RMV, 100) <= ULTRA_RMV_MAX: highlights.append("Ultra Low Volatility")
    elif row.get(TITLE_RMV, 100) <= CONTROLLED_RMV_MAX: highlights.append("Controlled Volatility")
    
    if not highlights:
        highlights.append("High Relative Strength")
    return ", ".join(highlights)

def _export_web_data(df_stocks, df_industries, quick=False, sector_stages=None):
    """Exports top 6 stocks and top 6 industries to web_data.json or quick_web_data.json"""
    # Select top 6 stocks: priority High-Win, then Flip, then RS
    top_stocks = []
    
    # 1. High Win
    high_win = df_stocks[df_stocks[TITLE_CAND_HIGH]].sort_values(TITLE_RS, ascending=False).head(6)
    top_stocks.extend(high_win.to_dict('records'))
    
    # 2. If < 6, add Flip
    if len(top_stocks) < 6:
        needed = 6 - len(top_stocks)
        existing_tickers = [s[TITLE_TICKER] for s in top_stocks]
        flips = df_stocks[df_stocks[TITLE_CAND_FLIP] & ~df_stocks[TITLE_TICKER].isin(existing_tickers)]
        flips = flips.sort_values(TITLE_RS, ascending=False).head(needed)
        top_stocks.extend(flips.to_dict('records'))
        
    # 3. If still < 6, add Top RS (ensuring low RMV)
    if len(top_stocks) < 6:
        needed = 6 - len(top_stocks)
        existing_tickers = [s[TITLE_TICKER] for s in top_stocks]
        top_rs = df_stocks[
            ~df_stocks[TITLE_TICKER].isin(existing_tickers) & 
            (df_stocks[TITLE_RMV] <= CONTROLLED_RMV_MAX)
        ]
        top_rs = top_rs.sort_values(TITLE_RS, ascending=False).head(needed)
        top_stocks.extend(top_rs.to_dict('records'))

    # Format stocks output
    formatted_stocks = []
    for i, s in enumerate(top_stocks):
        formatted_stocks.append({
            "rank": i + 1,
            "ticker": s[TITLE_TICKER],
            "name": s.get(TITLE_NAME, s[TITLE_TICKER]),
            "rs": int(s[TITLE_PERCENTILE]),
            "rs_raw": round(s[TITLE_RS], 2),
            "rs_1w_pct": int(s.get("rs_1w_pct", 50)),
            "rs_1m_pct": int(s.get("rs_1m_pct", 50)),
            "rmv": s[TITLE_RMV],
            "industry": s[TITLE_INDUSTRY],
            "sector": s[TITLE_SECTOR],
            "highlights": _get_highlights(s),
            "source": s.get(TITLE_SOURCE, "AI Scanner"),
            "canslim": s.get(TITLE_CANSLIM, {}),
            "days_to_earnings": int(s.get(TITLE_DTE, -1)),
            "is_restricted": bool(s.get("is_restricted", False)),
            "tradingview_url": f"https://www.tradingview.com/chart/?symbol={s.get(TITLE_TICKER, 'SPY')}",
            "finviz_chart_url": f"https://charts2.finviz.com/chart.ashx?t={s.get(TITLE_TICKER, 'SPY')}&ty=c&ta=0&p=d&s=l"
        })

    # Export all AI-screened stocks with RS >= 70 for client-side slider filtering
    filterable = df_stocks[
        (df_stocks[TITLE_PERCENTILE] >= 70) & 
        (df_stocks[TITLE_SOURCE] == "AI Scanner") &
        (df_stocks[TITLE_UNIVERSE] != "Market Pulse")
    ].sort_values(TITLE_RS, ascending=False)
    all_stocks_list = []
    for i, (_, s) in enumerate(filterable.iterrows()):
        all_stocks_list.append({
            "rank": i + 1,
            "ticker": s[TITLE_TICKER],
            "name": s.get(TITLE_NAME, s[TITLE_TICKER]),
            "rs": int(s[TITLE_PERCENTILE]),
            "rs_raw": round(s[TITLE_RS], 2),
            "rs_1w_pct": int(s.get("rs_1w_pct", 50)),
            "rs_1m_pct": int(s.get("rs_1m_pct", 50)),
            "rmv": s[TITLE_RMV],
            "industry": s[TITLE_INDUSTRY],
            "sector": s[TITLE_SECTOR],
            "highlights": _get_highlights(s),
            "source": s.get(TITLE_SOURCE, "AI Scanner"),
            "canslim": s.get(TITLE_CANSLIM, {}),
            "days_to_earnings": int(s.get(TITLE_DTE, -1)),
            "is_restricted": bool(s.get("is_restricted", False)),
            "tradingview_url": f"https://www.tradingview.com/chart/?symbol={s.get(TITLE_TICKER, 'SPY')}",
            "finviz_chart_url": f"https://charts2.finviz.com/chart.ashx?t={s.get(TITLE_TICKER, 'SPY')}&ty=c&ta=0&p=d&s=l"
        })

    # Format all manual tips separately for the dedicated "Tips" section
    manual_tips_df = df_stocks[df_stocks[TITLE_SOURCE] != "AI Scanner"]
    formatted_tips = []
    for _, s in manual_tips_df.iterrows():
        formatted_tips.append({
            "rank": "TIP",
            "ticker": s[TITLE_TICKER],
            "name": s.get(TITLE_NAME, s[TITLE_TICKER]),
            "rs": int(s[TITLE_PERCENTILE]),
            "rs_raw": round(s[TITLE_RS], 2),
            "rs_1w_pct": int(s.get("rs_1w_pct", 50)),
            "rs_1m_pct": int(s.get("rs_1m_pct", 50)),
            "rmv": s[TITLE_RMV],
            "industry": s[TITLE_INDUSTRY],
            "sector": s[TITLE_SECTOR],
            "highlights": _get_highlights(s),
            "source": s.get(TITLE_SOURCE, "Manual Tip"),
            "label": s.get("label", ""),
            "date": s.get("date", ""),
            "canslim": s.get(TITLE_CANSLIM, {}),
            "days_to_earnings": int(s.get(TITLE_DTE, -1)),
            "is_restricted": bool(s.get("is_restricted", False)),
            "tradingview_url": f"https://www.tradingview.com/chart/?symbol={s.get(TITLE_TICKER, 'SPY')}",
            "finviz_chart_url": f"https://charts2.finviz.com/chart.ashx?t={s.get(TITLE_TICKER, 'SPY')}&ty=c&ta=0&p=d&s=l"
        })

    # Format top 6 industries
    top_6_ind = df_industries.head(6)
    formatted_industries = []
    for i, (_, row) in enumerate(top_6_ind.iterrows()):
        formatted_industries.append({
            "rank": i + 1,
            "industry": row[TITLE_INDUSTRY],
            "sector": row[TITLE_SECTOR],
            "rs": round(row[TITLE_RS], 2),
            "tickers": row[TITLE_TICKERS].split(',')[:5] # Show top 5 tickers
        })

    # Format Market Pulse Section
    pulse_tickers = ["DIA", "SPY", "RSP", "QQQ", "QQQE", "MDY", "IWM"] # Markets at top
    pulse_df = df_stocks[df_stocks[TITLE_UNIVERSE] == "Market Pulse"]
    
    formatted_pulse = []
    
    # helper for sorting pulse
    def pulse_sort_order(r):
        if r[TITLE_TICKER] in pulse_tickers: return 10
        if r[TITLE_COMMENTARY] == "Upward": return 5
        if r[TITLE_COMMENTARY] == "Sideways": return 3
        return 1

    pulse_out_df = pulse_df.copy()
    pulse_out_df["prio"] = pulse_out_df.apply(pulse_sort_order, axis=1)
    pulse_out_df = pulse_out_df.sort_values("prio", ascending=False)

    for _, s in pulse_out_df.iterrows():
        formatted_pulse.append({
            "ticker": s[TITLE_TICKER],
            "name": s.get(TITLE_NAME, s[TITLE_TICKER]),
            "rs_rank": int(s[TITLE_PERCENTILE]),
            "trend": s.get(TITLE_COMMENTARY, "Sideways"),
            "commentary": s.get("Commentary_Text", "Analyzing trend..."),
            "tradingview_url": f"https://www.tradingview.com/chart/?symbol={s.get(TITLE_TICKER, 'SPY')}"
        })

    web_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stocks": formatted_stocks,
        "all_stocks": all_stocks_list,
        "tips": formatted_tips,
        "industries": formatted_industries,
        "pulse": formatted_pulse
    }

    # Add Stage Analysis if available
    formatted_stages = []
    if sector_stages:
        for s, counts in sector_stages.items():
            if not s or s == "unknown" or s == "--- Reference ---": continue
            total = counts["total"]
            if total == 0: continue
            
            s1_p = round((counts["s1"] / total) * 100)
            s2_p = round((counts["s2"] / total) * 100)
            s3_p = round((counts["s3"] / total) * 100)
            s4_p = round((counts["s4"] / total) * 100)
            
            health = "Neutral"
            if s2_p > 50: health = "Strong"
            elif s2_p > 30: health = "Healthy"
            elif s4_p > 40: health = "Weak"
            elif s1_p + s2_p > 50: health = "Improving"
            
            # Sort tickers by RS for display
            sorted_t = sorted(counts["tickers"], key=lambda x: x["rs"], reverse=True)
            ticker_list = [x["t"] for x in sorted_t]
            
            formatted_stages.append({
                "sector": s,
                "count": total,
                "s1": s1_p,
                "s2": s2_p,
                "s3": s3_p,
                "s4": s4_p,
                "tickers": ticker_list,
                "health": health
            })
        
        # Sort by Stage 2 Percentage
        formatted_stages = sorted(formatted_stages, key=lambda x: x["s2"], reverse=True)

    web_data["stages"] = formatted_stages

    filename = "quick_web_data.json" if quick else "web_data.json"
    output_path = os.path.join(DIR, "output", filename)
    with open(output_path, 'w') as f:
        json.dump(web_data, f, indent=4)
    print(f"Web data exported to {output_path}")

# -------------------------
# Main rankings
# -------------------------
def _process_single_ticker(ticker, ticker_data, ref_candles, spy_ok, minervini_stage2):
    """Worker function for parallel processing."""
    try:
        if ticker_data.get("skip_calc", 1) != 0:
            return None

        # Universe filters are handled in the main loop to avoid passing config
        candles = ticker_data["candles"]
        if not candles or len(candles) < 120:
            return None

        # Optimization: pre-calculate series
        closes_series = pd.Series([c["close"] for c in candles])
        if closes_series.iloc[-1] <= 12:
            return None

        highs_series = pd.Series([c["high"] for c in candles])
        lows_series = pd.Series([c["low"] for c in candles])
        closes_ref_series = pd.Series([c["close"] for i, c in enumerate(ref_candles) if i < len(candles)]) # Basic alignment

        rs = relative_strength(closes_series, closes_ref_series)
        rmv = calculate_rmv(closes_series, highs_series, lows_series)

        month = 20
        week = 5
        rs1w = relative_strength(closes_series.head(-week), closes_ref_series.head(-week))
        rs1m = relative_strength(closes_series.head(-month), closes_ref_series.head(-month))
        rs3m = relative_strength(closes_series.head(-3 * month), closes_ref_series.head(-3 * month))
        rs6m = relative_strength(closes_series.head(-6 * month), closes_ref_series.head(-6 * month))

        tdf = _to_df_from_candles(candles)
        if tdf is None or len(tdf) < 200:
            return None
        
        tdf = _compute_daily_features(tdf)
        wdf = _compute_weekly(tdf)
        if wdf is None or len(wdf) < 12:
            return None

        latest_daily = tdf.iloc[-1]
        latest_weekly = wdf.iloc[-1]
        ptc = _compute_ptc(latest_daily, latest_weekly)

        flip_setup = (
            ptc == 5 and
            (abs(latest_daily["Close"] - latest_daily["ma50"]) / latest_daily["ma50"] < 0.02) and
            (tdf["atr14"].iloc[-1] < tdf["atr14"].iloc[-6]) and
            (latest_daily["Volume"] < latest_daily["vol20"])
        )

        # Use industry/sector from ticker_data or metadata
        meta = TICKER_INFO_DICT.get(ticker, {}).get("info", {})
        industry = ticker_data.get("industry", meta.get("industry", "unknown"))
        sector = ticker_data.get("sector", meta.get("sector", "unknown"))
        universe = ticker_data.get("universe", "unknown")
        name = meta.get("name", ticker)

        # CANSLIM & Earnings
        eps_c = meta.get("eps_growth_curr", 0) > 0.20
        eps_a = meta.get("eps_growth_annual", 0) > 0.20
        
        high_52w = float(tdf["High"].tail(252).max())
        new_high = latest_daily["Close"] > (high_52w * 0.95)
        
        supply_demand = latest_daily["Volume"] > latest_daily["vol20"]
        
        # Days to earnings
        earnings_date_str = meta.get("earnings_date", "n/a")
        days_to_earnings = -1
        if earnings_date_str != "n/a":
            try:
                edate = datetime.strptime(earnings_date_str, '%Y-%m-%d')
                days_to_earnings = (edate - datetime.now()).days
            except:
                pass

        canslim = {
            "c": bool(eps_c),
            "a": bool(eps_a),
            "n": bool(new_high),
            "s": bool(supply_demand),
            "l": False, # Will be set in main rankings
            "i": rs > 1.2, # Proxy: RS > 1.2 is strong institutional support
            "m": bool(spy_ok)
        }

        status, trend_comment = _get_trend_commentary(tdf)
        stage = _get_standard_stage(tdf)

        return (
            ticker, minervini_stage2, sector, industry, universe,
            rs, 0, rs1w, rs1m, rs3m, rs6m, rmv,
            float(latest_daily["Close"]),
            float(latest_daily["atr14_pct"]) if pd.notna(latest_daily["atr14_pct"]) else np.nan,
            int(ptc),
            bool(latest_daily["contraction"]),
            bool(latest_daily["confirmed_breakout"]),
            bool(latest_daily["not_extended"]),
            bool(flip_setup),
            bool(spy_ok),
            ticker_data.get("source", "AI Scanner"),
            canslim,
            int(days_to_earnings),
            name,
            status,
            trend_comment,
            stage,
            ticker_data.get("label", ""),
            ticker_data.get("date", "")
        )
    except Exception as e:
        # print(f"Error processing {ticker}: {e}")
        return None

def rankings(test_mode=False, test_tickers=None, quick=False):
    """Returns a dataframe with percentile rankings for relative strength + signals."""
    json_data = read_json(PRICE_DATA)
    relative_strengths = []
    industries = {}
    stock_rs = {}

    ref = json_data[REFERENCE_TICKER]
    ref_df = _to_df_from_candles(ref["candles"])
    ref_df = _compute_daily_features(ref_df) if ref_df is not None else None
    spy_ok = _spy_trend_ok(ref_df) if ref_df is not None and len(ref_df) >= 210 else True

    if test_mode and test_tickers:
        tickers_to_process = [t for t in test_tickers if t in json_data]
    else:
        # Filter tickers by universe/config before parallelizing
        tickers_to_process = []
        for t in json_data.keys():
            # Always include manual tips
            source = json_data[t].get("source", "AI Scanner")
            if source != "AI Scanner":
                tickers_to_process.append(t)
                continue

            u = json_data[t].get("universe", "")
            if u == "S&P 500" and not cfg("SP500"): continue
            if u == "S&P 400" and not cfg("SP400"): continue
            if u == "S&P 600" and not cfg("SP600"): continue
            if u == "Nasdaq 100" and not cfg("NQ100"): continue
            tickers_to_process.append(t)

    print(f"Analyzing {len(tickers_to_process)} tickers in parallel...")
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _process_single_ticker, 
                t, json_data[t], ref["candles"], spy_ok, 
                json_data[t].get("minervini", 0)
            ): t for t in tickers_to_process if t != REFERENCE_TICKER
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    sector_stages = {}

    for res in results:
        ticker, mm, sector, industry, universe, rs, pct, rs1w, rs1m, rs3m, rs6m, rmv, close, atr, ptc, contr, brk, nextend, flip, sok, source, canslim, dte, name, status, comment, stage, label, date = res
        
        if sector not in sector_stages:
            sector_stages[sector] = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "total": 0, "tickers": []}
        sector_stages[sector][f"s{stage}"] += 1
        sector_stages[sector]["total"] += 1
        sector_stages[sector]["tickers"].append({"t": ticker, "rs": rs})
        
        if (industry == "n/a" or not industry):
            if universe == "Market Pulse":
                industry = "Index/ETF"  # Keep pulse tickers with a default industry
            else:
                continue
        is_restricted = industry in EXCLUDE_INDUSTRIES
        
        relative_strengths.append((
            0, ticker, mm, sector, industry, universe,
            rs, pct, rs1w, rs1m, rs3m, rs6m, rmv,
            close, atr, ptc, contr, brk, nextend, flip, sok, source, canslim, dte, is_restricted, name, status, comment, stage, label, date
        ))
        stock_rs[ticker] = rs

        if industry not in industries:
            industries[industry] = {
                "info": (0, industry, sector, 0, 99, 1, 3, 6),
                TITLE_RS: [], TITLE_1M: [], TITLE_3M: [], TITLE_6M: [], TITLE_TICKERS: []
            }
        industries[industry][TITLE_RS].append(rs)
        industries[industry][TITLE_1M].append(rs1m)
        industries[industry][TITLE_3M].append(rs3m)
        industries[industry][TITLE_6M].append(rs6m)
        industries[industry][TITLE_TICKERS].append(ticker)

    dfs = []
    suffix = ''

    df = pd.DataFrame(relative_strengths, columns=[
        TITLE_RANK, TITLE_TICKER, TITLE_MINERVINI, TITLE_SECTOR, TITLE_INDUSTRY, TITLE_UNIVERSE,
        TITLE_RS, TITLE_PERCENTILE, "RS_1W", TITLE_1M, TITLE_3M, TITLE_6M, TITLE_RMV,
        TITLE_CLOSE, TITLE_ATR_PCT, TITLE_PTC, TITLE_CONTRACTION, TITLE_BREAKOUT,
        TITLE_NOT_EXT, TITLE_FLIP, TITLE_SPY_OK, TITLE_SOURCE, TITLE_CANSLIM, TITLE_DTE, "is_restricted",
        TITLE_NAME, TITLE_COMMENTARY, "Commentary_Text", TITLE_STAGE, "label", "date"
    ])

    if df.empty:
        print("No stocks data to rank.")
        return dfs

    # Percentile bins
    df[TITLE_PERCENTILE] = pd.qcut(df[TITLE_RS], 100, precision=64, labels=False, duplicates='drop') + 1
    df["rs_1w_pct"] = pd.qcut(df["RS_1W"], 100, precision=64, labels=False, duplicates='drop') + 1
    df["rs_1m_pct"] = pd.qcut(df[TITLE_1M], 100, precision=64, labels=False, duplicates='drop') + 1
    df[TITLE_3M] = pd.qcut(df[TITLE_3M], 100, precision=64, labels=False, duplicates='drop') + 1
    df[TITLE_6M] = pd.qcut(df[TITLE_6M], 100, precision=64, labels=False, duplicates='drop') + 1

    # Finalize CANSLIM 'L'
    def _finalize_canslim(r):
        c = r[TITLE_CANSLIM]
        c["l"] = r[TITLE_PERCENTILE] >= 80
        return c
    df[TITLE_CANSLIM] = df.apply(_finalize_canslim, axis=1)

    df = df.sort_values(([TITLE_RS]), ascending=False)
    df[TITLE_RANK] = range(1, len(df) + 1)

    # trim to MIN_PERCENTILE (but keep ALL manual tips)
    # Keep stocks above MIN_PERCENTILE, manual tips, AND Market Pulse tickers
    df = df[(df[TITLE_PERCENTILE] >= MIN_PERCENTILE) | (df[TITLE_SOURCE] != "AI Scanner") | (df[TITLE_UNIVERSE] == "Market Pulse")]

    # Minervini subset
    dfm = df[df[TITLE_MINERVINI] > 6]
    dfm = dfm.sort_values(([TITLE_MINERVINI, TITLE_RANK]), ascending=[False, True])

    print(df.head())
    print(dfm.head())

    # --- Candidate flags ---
    def _is_controlled_universe(r):
        return r[TITLE_INDUSTRY] not in EXCLUDE_INDUSTRIES

    # High-win = confirmed breakout AFTER contraction + strong trend alignment
    df[TITLE_CAND_HIGH] = df.apply(lambda r: bool(
        _is_controlled_universe(r) and
        r[TITLE_SPY_OK] and
        r[TITLE_MINERVINI] >= 7 and
        r[TITLE_PERCENTILE] >= 97 and
        r[TITLE_RMV] <= CONTROLLED_RMV_MAX and
        r[TITLE_PTC] >= 6 and
        r[TITLE_CONTRACTION] and
        r[TITLE_BREAKOUT] and
        r[TITLE_NOT_EXT]
    ), axis=1)

    # Flip = early candidates likely to move PTC 5->6 soon (no breakout required)
    df[TITLE_CAND_FLIP] = df.apply(lambda r: bool(
        _is_controlled_universe(r) and
        r[TITLE_SPY_OK] and
        r[TITLE_MINERVINI] >= 7 and
        r[TITLE_PERCENTILE] >= 97 and
        r[TITLE_RMV] <= FLIP_RMV_MAX and
        r[TITLE_FLIP] and
        r[TITLE_NOT_EXT]
    ), axis=1)

    # --- Writes ---
    df.to_csv(os.path.join(DIR, "output", f'rs_stocks{suffix}.csv'), index=False)
    dfm.to_csv(os.path.join(DIR, "output", f'rs_stocks_minervini.csv'), index=False)

    # Full scored sheet
    df.to_csv(os.path.join(DIR, "output", "rs_stocks_scored.csv"), index=False)

    # Candidates
    df[df[TITLE_CAND_HIGH]].to_csv(os.path.join(DIR, "output", "rs_stocks_candidates_high_win.csv"), index=False)
    df[df[TITLE_CAND_FLIP]].to_csv(os.path.join(DIR, "output", "rs_stocks_candidates_flip.csv"), index=False)

    # Create low RMV list (calibrated to your RMV scale)
    df_low_rmv = df[df[TITLE_RMV] <= CONTROLLED_RMV_MAX].copy()
    if not df_low_rmv.empty:
        df_low_rmv = df_low_rmv.sort_values(TITLE_RMV, ascending=True)
        df_low_rmv[TITLE_RANK] = range(1, len(df_low_rmv) + 1)
        df_low_rmv.to_csv(os.path.join(DIR, "output", "rs_stocks_low_rmv.csv"), index=False)
        print(f"\nLow RMV stocks (RMV <= {CONTROLLED_RMV_MAX}): {len(df_low_rmv)} stocks")
        print(df_low_rmv[[TITLE_RANK, TITLE_TICKER, TITLE_RMV, TITLE_RS, TITLE_SECTOR, TITLE_INDUSTRY]].head(10))
    else:
        print(f"\nNo stocks found with RMV <= {CONTROLLED_RMV_MAX}")

    # Also create a list showing all stocks sorted by RMV
    df_sorted_rmv = df.sort_values(TITLE_RMV, ascending=True).copy()
    df_sorted_rmv[TITLE_RANK] = range(1, len(df_sorted_rmv) + 1)
    df_sorted_rmv.to_csv(os.path.join(DIR, "output", "rs_stocks_by_rmv.csv"), index=False)
    print(f"\nAll stocks sorted by RMV (lowest first): {len(df_sorted_rmv)} stocks")
    print(df_sorted_rmv[[TITLE_RANK, TITLE_TICKER, TITLE_RMV, TITLE_RS, TITLE_SECTOR, TITLE_INDUSTRY]].head(10))

    # Create rmv_rs.csv filtered for RMV <= CONTROLLED_RMV_MAX
    df_rmv_rs = df[df[TITLE_RMV] <= CONTROLLED_RMV_MAX].copy()
    if not df_rmv_rs.empty:
        df_rmv_rs = df_rmv_rs.sort_values(TITLE_RMV, ascending=True)
        df_rmv_rs_output = df_rmv_rs[[TITLE_RMV, TITLE_TICKER, TITLE_MINERVINI, TITLE_SECTOR, TITLE_INDUSTRY,
                                     TITLE_PERCENTILE, TITLE_RS, TITLE_1M]].copy()
        df_rmv_rs_output.to_csv(os.path.join(DIR, "output", "rmv_rs.csv"), index=False)
        print(f"\nRMV-RS list (RMV <= {CONTROLLED_RMV_MAX}): {len(df_rmv_rs_output)} stocks")
        print(df_rmv_rs_output.head(10))
    else:
        print(f"\nNo stocks found with RMV <= {CONTROLLED_RMV_MAX} for rmv_rs.csv")

    # Minervini_list.csv
    try:
        list_of_dfm_tickers = dfm[TITLE_TICKER].to_list()
        list_of_dfm_tickers = ", ".join(map(str, list_of_dfm_tickers))
        with open(os.path.join(DIR, "output", "Minervini_list.csv"), 'w') as f:
            f.write(list_of_dfm_tickers)
    except:
        print('Minervini_list not created')

    # -------------------------
    # Industries analysis (kept from your original)
    # -------------------------
    def getDfView(industry_entry):
        return industry_entry["info"]

    def sum_fn(a, b):
        return a + b

    def getRsAverage(industries_dict, industry_name, column):
        rs_vals = industries_dict[industry_name][column]
        rs_avg = reduce(sum_fn, rs_vals) / len(rs_vals)
        rs_avg = int(rs_avg * 100) / 100
        return rs_avg

    def rs_for_stock(t):
        return stock_rs[t]

    def getTickers(industries_dict, industry_name):
        return ",".join(sorted(industries_dict[industry_name][TITLE_TICKERS], key=rs_for_stock, reverse=True))

    # Web Dashboard Export (always run before potential early return)
    _export_web_data(df, pd.DataFrame(columns=[TITLE_INDUSTRY, TITLE_SECTOR, TITLE_RS, TITLE_TICKERS]))

    # remove industries with only one stock
    filtered_industries = filter(lambda i: len(i[TITLE_TICKERS]) > 1, list(industries.values()))
    filtered_industries_list = list(filtered_industries)

    if len(filtered_industries_list) == 0:
        print("No industries with multiple stocks found. Skipping industry analysis.")
        return [df]

    df_industries = pd.DataFrame(map(getDfView, filtered_industries_list),
                                columns=[TITLE_RANK, TITLE_INDUSTRY, TITLE_SECTOR, TITLE_RS, TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M])

    df_industries[TITLE_RS] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_RS), axis=1)
    df_industries[TITLE_1M] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_1M), axis=1)
    df_industries[TITLE_3M] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_3M), axis=1)
    df_industries[TITLE_6M] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_6M), axis=1)

    df_industries[TITLE_PERCENTILE] = pd.qcut(df_industries[TITLE_RS], 100, labels=False, duplicates="drop")
    df_industries[TITLE_1M] = pd.qcut(df_industries[TITLE_1M], 100, labels=False, duplicates="drop")
    df_industries[TITLE_3M] = pd.qcut(df_industries[TITLE_3M], 100, labels=False, duplicates="drop")
    df_industries[TITLE_6M] = pd.qcut(df_industries[TITLE_6M], 100, labels=False, duplicates="drop")
    df_industries[TITLE_TICKERS] = df_industries.apply(lambda row: getTickers(industries, row[TITLE_INDUSTRY]), axis=1)

    df_industries = df_industries.sort_values(([TITLE_PERCENTILE]), ascending=False)

    df_industries[TITLE_RANK] = range(1, len(df_industries) + 1)

    df_industries.to_csv(os.path.join(DIR, "output", f'rs_industries{suffix}.csv'), index=False)

    # Final Web Dashboard Export (refreshed with industries if they exist)
    _export_web_data(df, df_industries, quick=quick, sector_stages=sector_stages)

    return [df, df_industries]

    return [df, df_industries]


def main(skipEnter_legacy=None):
    parser = argparse.ArgumentParser(description='RS Ranking Engine')
    parser.add_argument('--quick', action='store_true', help='Quick scan mode (outputs to quick_web_data.json)')
    parser.add_argument('--test', action='store_true', help='Test mode')
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 and not (isinstance(skipEnter_legacy, bool) or skipEnter_legacy is not None) else [])

    # Support legacy positional arguments
    is_test = args.test
    is_quick = args.quick

    if args.test:
        print("Running in TEST MODE")
        ranks = rankings(test_mode=True, quick=args.quick)
    else:
        ranks = rankings(quick=args.quick)

    if ranks:
        print(ranks[0].head(20))
        print(f"***\nDone. Final export to {'quick_web_data.json' if args.quick else 'web_data.json'}.\n***")
    else:
        print("No data processed.")

    if cfg("EXIT_WAIT_FOR_ENTER"):
        input("Press Enter key to exit...")


if __name__ == "__main__":
    main()
