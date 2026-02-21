import sys
import pandas as pd
import numpy as np
import json
import os
import csv
import yaml
from functools import reduce

from rs_data import cfg, read_json as read_json_from_rs_data

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
    with open(json_file, "r") as fp:
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


# -------------------------
# Main rankings
# -------------------------
def rankings(test_mode=False, test_tickers=None):
    """Returns a dataframe with percentile rankings for relative strength + signals."""
    json_data = read_json(PRICE_DATA)
    relative_strengths = []
    ranks = []
    industries = {}
    ind_ranks = []
    stock_rs = {}

    ref = json_data[REFERENCE_TICKER]

    # SPY regime filter from reference ticker candles
    ref_df = _to_df_from_candles(ref["candles"])
    ref_df = _compute_daily_features(ref_df) if ref_df is not None else None
    spy_ok = _spy_trend_ok(ref_df) if ref_df is not None and len(ref_df) >= 210 else True

    # Test mode subset
    if test_mode and test_tickers:
        tickers_to_process = [t for t in test_tickers if t in json_data]
        print(f"Test mode: Processing {len(tickers_to_process)} tickers: {tickers_to_process}")
    else:
        tickers_to_process = json_data.keys()

    for ticker in tickers_to_process:
        try:
            if json_data[ticker].get("skip_calc", 1) != 0:
                continue

            # Universe filters
            if not cfg("SP500") and json_data[ticker]["universe"] == "S&P 500":
                continue
            if not cfg("SP400") and json_data[ticker]["universe"] == "S&P 400":
                continue
            if not cfg("SP600") and json_data[ticker]["universe"] == "S&P 600":
                continue
            if not cfg("NQ100") and json_data[ticker]["universe"] == "Nasdaq 100":
                continue

            candles = json_data[ticker]["candles"]
            if not candles or len(candles) < 6 * 20:
                continue

            closes = list(map(lambda c: c["close"], candles))
            highs = list(map(lambda c: c["high"], candles))
            lows = list(map(lambda c: c["low"], candles))
            closes_ref = list(map(lambda c: c["close"], ref["candles"]))

            industry = TICKER_INFO_DICT[ticker]["info"]["industry"] if json_data[ticker]["industry"] == "unknown" else json_data[ticker]["industry"]
            sector = TICKER_INFO_DICT[ticker]["info"]["sector"] if json_data[ticker]["sector"] == "unknown" else json_data[ticker]["sector"]

            if industry == "n/a" or len(str(industry).strip()) == 0:
                continue

            closes_series = pd.Series(closes)
            highs_series = pd.Series(highs)
            lows_series = pd.Series(lows)
            closes_ref_series = pd.Series(closes_ref)

            rs = relative_strength(closes_series, closes_ref_series)
            rmv = calculate_rmv(closes_series, highs_series, lows_series)

            month = 20
            tmp_percentile = 100
            rs1m = relative_strength(closes_series.head(-1 * month), closes_ref_series.head(-1 * month))
            rs3m = relative_strength(closes_series.head(-3 * month), closes_ref_series.head(-3 * month))
            rs6m = relative_strength(closes_series.head(-6 * month), closes_ref_series.head(-6 * month))

            # Basic price filter
            if closes_series.iloc[-1] <= 12:
                continue

            # Compute signals from candle dataframe (needs datetime)
            tdf = _to_df_from_candles(candles)
            if tdf is None or len(tdf) < 220:
                continue
            tdf = _compute_daily_features(tdf)
            wdf = _compute_weekly(tdf)
            if wdf is None or len(wdf) < 12:
                continue

            latest_daily = tdf.iloc[-1]
            latest_weekly = wdf.iloc[-1]
            ptc = _compute_ptc(latest_daily, latest_weekly)

            flip_setup = (
                ptc == 5 and
                (abs(latest_daily["Close"] - latest_daily["ma50"]) / latest_daily["ma50"] < 0.02) and
                (tdf["atr14"].iloc[-1] < tdf["atr14"].iloc[-6]) and
                (latest_daily["Volume"] < latest_daily["vol20"])
            )

            # stocks output
            ranks.append(len(ranks) + 1)
            relative_strengths.append((
                0, ticker, json_data[ticker]["minervini"], sector, industry, json_data[ticker]["universe"],
                rs, tmp_percentile, rs1m, rs3m, rs6m, rmv,
                float(latest_daily["Close"]),
                float(latest_daily["atr14_pct"]) if pd.notna(latest_daily["atr14_pct"]) else np.nan,
                int(ptc),
                bool(latest_daily["contraction"]),
                bool(latest_daily["confirmed_breakout"]),
                bool(latest_daily["not_extended"]),
                bool(flip_setup),
                bool(spy_ok),
            ))
            stock_rs[ticker] = rs

            # industries output
            if industry not in industries:
                industries[industry] = {
                    "info": (0, industry, sector, 0, 99, 1, 3, 6),
                    TITLE_RS: [],
                    TITLE_1M: [],
                    TITLE_3M: [],
                    TITLE_6M: [],
                    TITLE_TICKERS: []
                }
                ind_ranks.append(len(ind_ranks) + 1)

            industries[industry][TITLE_RS].append(rs)
            industries[industry][TITLE_1M].append(rs1m)
            industries[industry][TITLE_3M].append(rs3m)
            industries[industry][TITLE_6M].append(rs6m)
            industries[industry][TITLE_TICKERS].append(ticker)

        except Exception:
            print(f'Ticker {ticker} has corrupted data.')
            continue

    dfs = []
    suffix = ''

    df = pd.DataFrame(relative_strengths, columns=[
        TITLE_RANK, TITLE_TICKER, TITLE_MINERVINI, TITLE_SECTOR, TITLE_INDUSTRY, TITLE_UNIVERSE,
        TITLE_RS, TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M, TITLE_RMV,
        TITLE_CLOSE, TITLE_ATR_PCT, TITLE_PTC, TITLE_CONTRACTION, TITLE_BREAKOUT,
        TITLE_NOT_EXT, TITLE_FLIP, TITLE_SPY_OK
    ])

    if df.empty:
        print("No stocks data to rank.")
        return dfs

    # Percentile bins
    df[TITLE_PERCENTILE] = pd.qcut(df[TITLE_RS], 100, precision=64, labels=False, duplicates='drop')
    df[TITLE_1M] = pd.qcut(df[TITLE_1M], 100, precision=64, labels=False, duplicates='drop')
    df[TITLE_3M] = pd.qcut(df[TITLE_3M], 100, precision=64, labels=False, duplicates='drop')
    df[TITLE_6M] = pd.qcut(df[TITLE_6M], 100, precision=64, labels=False, duplicates='drop')

    df = df.sort_values(([TITLE_RS]), ascending=False)
    df[TITLE_RANK] = ranks

    # trim to MIN_PERCENTILE
    out_tickers_count = 0
    for _, row in df.iterrows():
        if row[TITLE_PERCENTILE] >= MIN_PERCENTILE:
            out_tickers_count += 1
    df = df.head(out_tickers_count)

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

    ind_ranks = ind_ranks[:len(df_industries)]
    df_industries[TITLE_RANK] = ind_ranks

    df_industries.to_csv(os.path.join(DIR, "output", f'rs_industries{suffix}.csv'), index=False)

    return [df, df_industries]


def main(skipEnter=False, test_mode=False, test_tickers=None):
    if test_mode:
        print("Running in TEST MODE")
        ranks = rankings(test_mode=True, test_tickers=test_tickers)
    else:
        ranks = rankings()

    if ranks:
        print(ranks[0].head(20))
        print("***\nYour 'rs_stocks.csv' is in the output folder.\n***")
    else:
        print("No data processed.")

    if not skipEnter and cfg("EXIT_WAIT_FOR_ENTER"):
        input("Press Enter key to exit...")


if __name__ == "__main__":
    main()
