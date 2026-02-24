#!/usr/bin/env python
import requests
import json
import time
import random
import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
#import pandas_datareader as pdr
import pickle
import requests
import yaml
try:
    import yfinance as yf
except ImportError as e:
    print(f"Error importing yfinance: {e}")
    print("This may be due to Python version compatibility issues.")
    print("Please ensure you're using Python 3.9+ or install compatible versions.")
    raise
import pandas as pd
import dateutil.relativedelta
import numpy as np
import re
from ftplib import FTP
from yahoo_fin.stock_info import get_quote_table
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from time import sleep
import sys
import argparse

from datetime import date
from datetime import datetime


DIR = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(DIR, 'data')):
    os.makedirs(os.path.join(DIR, 'data'))
if not os.path.exists(os.path.join(DIR, 'tmp')):
    os.makedirs(os.path.join(DIR, 'tmp'))

try:
    with open(os.path.join(DIR, 'config_private.yaml'), 'r') as stream:
        private_config = yaml.safe_load(stream)
except FileNotFoundError:
    private_config = None
except yaml.YAMLError as exc:
        print(exc)

try:
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
except FileNotFoundError:
    config = None
except yaml.YAMLError as exc:
        print(exc)

def cfg(key):
    try:
        return private_config[key]
    except:
        try:
            return config[key]
        except:
            return None

API_KEY = cfg("API_KEY")
TD_API = "https://api.tdameritrade.com/v1/marketdata/%s/pricehistory"
PRICE_DATA_OUTPUT = os.path.join(DIR, "data", "price_history.json")
REFERENCE_TICKER = cfg("REFERENCE_TICKER")
DATA_SOURCE = cfg("DATA_SOURCE")

def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as fp:
    #with open(json_file, "r") as fp:
        return json.load(fp)
        #return json.loads(fp.read())

API_KEY = cfg("API_KEY")
TD_API = "https://api.tdameritrade.com/v1/marketdata/%s/pricehistory"
PRICE_DATA_FILE = os.path.join(DIR, "data", "price_history.json")
REFERENCE_TICKER = cfg("REFERENCE_TICKER")
DATA_SOURCE = cfg("DATA_SOURCE")
ALL_STOCKS = cfg("USE_ALL_LISTED_STOCKS")
TICKER_INFO_FILE = os.path.join(DIR, "data_persist", "ticker_info.json")
TICKER_INFO_DICT = read_json(TICKER_INFO_FILE)



REF_TICKER = {"ticker": REFERENCE_TICKER, "sector": "--- Reference ---", "industry": "--- Reference ---", "universe": "--- Reference ---"}


UNKNOWN = "unknown"

PULSE_TICKERS = [
    # Markets
    "DIA", "SPY", "RSP", "QQQ", "QQQE", "MDY", "IWM",
    # Key Sectors
    "XLF", "SMH", "IYT", "XTN", "ITB", "XHB", "MAGS", "IBIT", "ETHA",
    "SLV", "GDX", "TAN", "GPN", "IBUY", "XLK", "XLU", "FDN", "USO", "XLE", "IGV"
]

def get_securities(url, ticker_pos = 1, table_pos = 1, sector_offset = 1, industry_offset = 1, universe = "N/A"):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        tables = soup.find_all('table', {'class': 'wikitable sortable'})
        if not tables or len(tables) < table_pos:
            return {}
        table = tables[table_pos-1]
        secs = {}
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) < ticker_pos: continue
            ticker = cells[ticker_pos-1].text.strip().split('\n')[0]
            sec = {"ticker": ticker, "universe": universe}
            try:
                sec["sector"] = cells[ticker_pos-1+sector_offset].text.strip()
                sec["industry"] = cells[ticker_pos-1+sector_offset+industry_offset].text.strip()
            except:
                sec["sector"] = UNKNOWN
                sec["industry"] = UNKNOWN
            secs[ticker] = sec
        return secs
    except Exception as e:
        print(f"Scraper error for {url}: {e}")
        return {}


def get_resolved_securities(full_scan=None):
    tickers = {REFERENCE_TICKER: REF_TICKER}
    use_full = full_scan if full_scan is not None else ALL_STOCKS
    if use_full:
        return get_tickers_from_nasdaq(tickers)
    else:
        try:
            return get_tickers_from_wikipedia(tickers)
        except Exception as e:
            print(f"Error fetching tickers from Wikipedia: {e}")
            print("Falling back to test ticker set...")
            # Use a small test set as fallback
            test_tickers = {
                "AAPL": {"ticker": "AAPL", "sector": "Technology", "industry": "Consumer Electronics", "universe": "NASDAQ"},
                "MSFT": {"ticker": "MSFT", "sector": "Technology", "industry": "Software", "universe": "NASDAQ"},
                "GOOGL": {"ticker": "GOOGL", "sector": "Technology", "industry": "Internet", "universe": "NASDAQ"},
                "AMZN": {"ticker": "AMZN", "sector": "Consumer Discretionary", "industry": "E-commerce", "universe": "NASDAQ"},
                "TSLA": {"ticker": "TSLA", "sector": "Consumer Discretionary", "industry": "Automotive", "universe": "NASDAQ"}
            }
            tickers.update(test_tickers)
            return tickers

def get_tickers_from_wikipedia(tickers):
    if cfg("NQ100"):
        # Nasdaq 100 is usually the first wikitable. table_pos=1.
        nq_secs = get_securities('https://en.wikipedia.org/wiki/Nasdaq-100', 1, 1, universe="Nasdaq 100")
        if not nq_secs:
            print("Using fallback Nasdaq 100 list...")
            fallback = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "COST", "PEP"]
            for t in fallback:
                tickers[t] = {"ticker": t, "sector": UNKNOWN, "industry": UNKNOWN, "universe": "Nasdaq 100"}
        else:
            tickers.update(nq_secs)
    if cfg("SP500"):
        tickers.update(get_securities('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies', sector_offset=3, universe="S&P 500"))
    if cfg("SP400"):
        tickers.update(get_securities('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', 2, universe="S&P 400"))
    if cfg("SP600"):
        tickers.update(get_securities('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', 2, universe="S&P 600"))
    return tickers

def load_manual_tips():
    """Loads manual tips from tips.yaml."""
    tips_file = os.path.join(DIR, "tips.yaml")
    if not os.path.exists(tips_file):
        return []
    try:
        with open(tips_file, 'r') as f:
            tips = yaml.safe_load(f)
            if not tips:
                return []
            return tips
    except Exception as e:
        print(f"Error loading tips.yaml: {e}")
        return []

def exchange_from_symbol(symbol):
    if symbol == "Q":
        return "NASDAQ"
    if symbol == "A":
        return "NYSE MKT"
    if symbol == "N":
       return "NYSE"
    if symbol == "P":
       return "NYSE ARCA"
    if symbol == "Z":
       return "BATS"
    if symbol == "V":
       return "IEXG"
    return "n/a"

def get_tickers_from_nasdaq(tickers):
    filename = "nasdaqtraded.txt"
    ticker_column = 1
    name_column = 2
    etf_column = 5
    exchange_column = 3
    test_column = 7
    ftp = FTP('ftp.nasdaqtrader.com')
    ftp.login()
    ftp.cwd('SymbolDirectory')
    lines = StringIO()
    ftp.retrlines('RETR '+filename, lambda x: lines.write(str(x)+'\n'))
    ftp.quit()
    lines.seek(0)
    results = lines.readlines()

    # Non-equity security types to exclude
    JUNK_KEYWORDS = {'warrant', 'warrants', 'rights', 'units', 'unit', 
                     'notes', 'debenture', 'debentures', 'bond',
                     'preferred'}
    skipped = 0

    for entry in results:
        sec = {}
        values = entry.split('|')
        if len(values) <= name_column:
            continue
            
        ticker = values[ticker_column]
        if re.match(r'^[A-Z]+$', ticker) and values[etf_column] == "N" and values[test_column] == "N":
            # Filter by Security Name — skip warrants, units, rights, debt, preferred
            full_name = values[name_column]
            sec_name_lower = full_name.lower()
            if any(kw in sec_name_lower for kw in JUNK_KEYWORDS):
                skipped += 1
                continue

            sec["ticker"] = ticker
            sec["name"] = full_name
            sec["sector"] = UNKNOWN
            sec["industry"] = UNKNOWN
            sec["universe"] = exchange_from_symbol(values[exchange_column])
            tickers[sec["ticker"]] = sec

    print(f"Loaded {len(tickers)} equities from NASDAQ (skipped {skipped} warrants/units/rights/debt/preferred)")
    return tickers

# SECURITIES = get_resolved_securities().values()  # Now handled in main()

def write_to_file(dict, file):
    with open(file, "w", encoding='utf8') as fp:
        json.dump(dict, fp, ensure_ascii=False)

def write_price_history_file(tickers_dict):
    write_to_file(tickers_dict, PRICE_DATA_FILE)

def write_ticker_info_file(info_dict):
    write_to_file(info_dict, TICKER_INFO_FILE)


def create_price_history_file(tickers_dict):
    with open(PRICE_DATA_OUTPUT, "w") as fp:
        json.dump(tickers_dict, fp)

def enrich_ticker_data(ticker_response, security, skip_calc, mm_count):
    ticker_response["name"] = security.get("name", security["ticker"])
    ticker_response["sector"] = security["sector"]
    ticker_response["industry"] = security["industry"]
    ticker_response["universe"] = security["universe"]
    ticker_response["source"] = security.get("source", "AI Scanner")
    ticker_response["label"] = security.get("label", "")
    ticker_response["date"] = security.get("date", "")
    ticker_response["skip_calc"] = skip_calc
    ticker_response["minervini"] = int(mm_count)
 

def tda_params(apikey, period_type="year", period=2, frequency_type="daily", frequency=1):
    """Returns tuple of api get params. Uses clenow default values."""
    return (
           ("apikey", apikey),
           ("periodType", period_type),
           ("period", period),
           ("frequencyType", frequency_type),
           ("frequency", frequency)
    )

def print_data_progress(ticker, universe, idx, securities, error_text, elapsed_s, remaining_s):
    dt_ref = datetime.fromtimestamp(0)
    dt_e = datetime.fromtimestamp(elapsed_s)
    elapsed = dateutil.relativedelta.relativedelta (dt_e, dt_ref)
    if remaining_s and not np.isnan(remaining_s):
        dt_r = datetime.fromtimestamp(remaining_s)
        remaining = dateutil.relativedelta.relativedelta (dt_r, dt_ref)
        remaining_string = f'{remaining.minutes}m {remaining.seconds}s'
    else:
        remaining_string = "?"
    print(f'{ticker} from {universe}{error_text} ({idx+1} / {len(securities)}). Elapsed: {elapsed.minutes}m {elapsed.seconds}s. Remaining: {remaining_string}.')

def get_remaining_seconds(all_load_times, idx, len):
    load_time_ma = pd.Series(all_load_times).rolling(np.minimum(idx+1, 25)).mean().tail(1).item()
    remaining_seconds = (len - idx) * load_time_ma
    return remaining_seconds

def escape_ticker(ticker):
    return ticker.replace(".","-")

def get_info_from_dict(dict, key):
    value = dict[key] if key in dict else "n/a"
    # fix unicode
    # value = value.replace("\u2014", " ")
    return value

def _fetch_single_ticker_info(ticker, max_retries=3):
    """Fetch info for a single ticker with exponential backoff."""
    escaped = escape_ticker(ticker)
    for attempt in range(max_retries):
        try:
            yt = yf.Ticker(escaped)
            info = yt.info

            if not info or "shortName" not in info:
                return ticker, {
                    "info": {
                        "name": ticker,
                        "industry": "Index/ETF",
                        "sector": "Market Pulse",
                        "marketCap": 0
                    }
                }

            return ticker, {
                "info": {
                    "name": info.get("longName", info.get("shortName", ticker)),
                    "industry": info.get("industry", "Index/ETF" if "^" in ticker else "unknown"),
                    "sector": info.get("sector", "Market Pulse" if "^" in ticker else "unknown"),
                    "marketCap": info.get("marketCap", 0),
                    # CANSLIM screening fields
                    "eps_growth_curr": info.get("earningsQuarterlyGrowth", 0) or 0,
                    "revenue_growth": info.get("revenueGrowth", 0) or 0,
                    "roe": info.get("returnOnEquity", 0) or 0,
                    "avg_volume": info.get("averageDailyVolume10Day", 0) or 0,
                    "current_price": info.get("currentPrice", 0) or 0,
                    # Quality / speculative screening fields
                    "trailing_eps": info.get("trailingEps", 0) or 0,
                    "operating_margin": info.get("operatingMargins", 0) or 0,
                    "debt_to_equity": info.get("debtToEquity", 999) if info.get("debtToEquity") is not None else 999,
                    "current_ratio": info.get("currentRatio", 0) or 0,
                    "free_cash_flow": info.get("freeCashflow", 0) or 0,
                    "ps_ratio": info.get("priceToSalesTrailing12Months", 0) or 0,
                    "pb_ratio": info.get("priceToBook", 0) or 0,
                }
            }
        except Exception as e:
            err_str = str(e)
            if "Too Many Requests" in err_str or "429" in err_str:
                wait = (5 * (3 ** attempt)) + random.uniform(0, 2)
                print(f"  Rate limited on {ticker}, backing off {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"  Error fetching info for {ticker}: {e}")
                return ticker, {"info": {"name": ticker, "industry": "unknown", "sector": "unknown", "marketCap": 0}}
    print(f"  Failed to fetch {ticker} after {max_retries} retries")
    return ticker, {"info": {"name": ticker, "industry": "unknown", "sector": "unknown", "marketCap": 0}}

def load_ticker_info_batch(tickers, current_info_dict):
    """Fetches metadata with 2-worker parallelism and aggressive backoff on 429."""
    if not tickers:
        return

    needed = []
    for t in tickers:
        info_block = current_info_dict.get(t, {}).get("info", {})
        name = info_block.get("name", "")
        # If name is missing OR is just the ticker symbol, we need a refetch attempt
        if not name or name == t:
            needed.append(t)
            
    if not needed:
        return

    print(f"Fetching metadata for {len(needed)} tickers (2 workers, backoff 5s/15s/45s)...")
    BATCH_SIZE = 25

    for batch_start in range(0, len(needed), BATCH_SIZE):
        batch = needed[batch_start:batch_start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(_fetch_single_ticker_info, t): t for t in batch}
            for future in as_completed(futures):
                try:
                    ticker, result = future.result()
                    current_info_dict[ticker] = result
                except Exception as e:
                    t = futures[future]
                    current_info_dict[t] = {"info": {"name": t, "industry": "unknown", "sector": "unknown", "marketCap": 0}}

        done = min(batch_start + BATCH_SIZE, len(needed))
        print(f"  Processed {done}/{len(needed)} tickers...")
        if done < len(needed):
            time.sleep(2)  # 2s pause between batches


def load_prices_from_tda(securities, api_key):
    print("*** Loading Stocks from TD Ameritrade ***")
    headers = {"Cache-Control" : "no-cache"}
    params = tda_params(api_key)
    tickers_dict = {}
    start = time.time()
    load_times = []

    for idx, sec in enumerate(securities):
        ticker = sec["ticker"]
        r_start = time.time()
        response = requests.get(
                TD_API % ticker,
                params=params,
                headers=headers
        )
        ticker_data = response.json()
        if not ticker in TICKER_INFO_DICT:
            new_entries = new_entries + 1
            load_ticker_info(ticker, TICKER_INFO_DICT)
            if new_entries % 25 == 0:
                write_ticker_info_file(TICKER_INFO_DICT)
        ticker_data["industry"] = TICKER_INFO_DICT[ticker]["info"]["industry"]
        now = time.time()
        current_load_time = now - r_start
        load_times.append(current_load_time)
        remaining_seconds = get_remaining_seconds(load_times, idx, len(securities))
        enrich_ticker_data(ticker_data, sec)
        tickers_dict[sec["ticker"]] = ticker_data
        error_text = f' Error with code {response.status_code}' if response.status_code != 200 else ''
        print_data_progress(sec["ticker"], sec["universe"], idx, securities, error_text, now - start, remaining_seconds)

    write_price_history_file(tickers_dict)

def convert_string_to_numeric(value_str):
    if not isinstance(value_str, str):
        # If value_str is not a string, assume it's already a numeric value
        try:
            numeric_value = float(value_str)
            return numeric_value
        except ValueError:
            print(f"Error: Unable to convert '{value_str}' to a numeric value.")
            return None

    value_str = value_str.upper()

    multipliers = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
        'BN': 1_000_000_000,
        'BILLION': 1_000_000_000,
        'T': 1_000_000_000_000,
    }

    for key, multiplier in multipliers.items():
        if key in value_str:
            numeric_value = float(value_str.replace(key, '').strip()) * multiplier
            return numeric_value

    try:
        numeric_value = float(value_str)
        return numeric_value
    except ValueError:
        print(f"Error: Unable to convert '{value_str}' to a numeric value.")
        return None






 







   
   


def get_yf_data(security, start_date, end_date):
    """
    Fetches data for a single security. 
    Note: For better performance, use batch downloading in load_prices_from_yahoo instead.
    """
    try:
        ticker = security["ticker"]
        escaped_ticker = escape_ticker(ticker)
        df = yf.download(escaped_ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        return _process_yf_df(df, security)
    except Exception as e:
        print(f"Error downloading data for {security.get('ticker')}: {e}")
        return _empty_ticker_data(security)

def _process_yf_df(df, security):
    ticker_data = {}
    if df.empty:
        return _empty_ticker_data(security)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return _empty_ticker_data(security)

    # Minervini Criteria (Optimized)
    try:
        c = df['Close']
        sma50 = c.rolling(50).mean().iloc[-1]
        sma150 = c.rolling(150).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        sma200_22 = c.rolling(200).mean().iloc[-22] if len(c) >= 222 else 0
        low_52w = c.tail(252).min()
        high_52w = c.tail(252).max()
        price = c.iloc[-1]

        mm_count = 0
        if price > sma150 and price > sma200: mm_count += 1
        if sma150 > sma200: mm_count += 1
        if sma200 > sma200_22: mm_count += 1
        if sma50 > sma150 and sma50 > sma200: mm_count += 1
        if price > sma50: mm_count += 1
        if price > 1.25 * low_52w: mm_count += 1
        if price >= 0.75 * high_52w: mm_count += 1
        mm_count += 1 # Base RS criterion
    except:
        mm_count = 0

    # Vectorized conversion to list of dicts
    df_candles = df[required_columns].copy()
    df_candles['datetime'] = df_candles.index.map(lambda x: int(x.timestamp()))
    
    df_dict = df_candles.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    
    ticker_data["candles"] = df_dict.to_dict('records')
    enrich_ticker_data(ticker_data, security, skip_calc=0, mm_count=mm_count)
    return ticker_data

def _empty_ticker_data(security):
    ticker_data = {"candles": []}
    enrich_ticker_data(ticker_data, security, skip_calc=1, mm_count=0)
    return ticker_data

def load_prices_from_yahoo(securities, info = {}):
    print("*** Loading Stocks from Yahoo Finance (Batch mode) ***")
    today = date.today()
    start = time.time()
    start_date = today - dt.timedelta(days=1*365+183) # 18 months
    
    tickers_dict = {}
    all_tickers = [sec["ticker"] for sec in securities]
    escaped_tickers = [escape_ticker(t) for t in all_tickers]
    
    # Chunked batch download — Yahoo drops results for oversized batches
    DOWNLOAD_CHUNK = 500
    print(f"Downloading price data for {len(escaped_tickers)} tickers in chunks of {DOWNLOAD_CHUNK}...")
    full_df = None
    
    for chunk_start in range(0, len(escaped_tickers), DOWNLOAD_CHUNK):
        chunk = escaped_tickers[chunk_start:chunk_start + DOWNLOAD_CHUNK]
        chunk_num = chunk_start // DOWNLOAD_CHUNK + 1
        total_chunks = (len(escaped_tickers) + DOWNLOAD_CHUNK - 1) // DOWNLOAD_CHUNK
        print(f"  Batch {chunk_num}/{total_chunks}: downloading {len(chunk)} tickers...")
        try:
            chunk_df = yf.download(chunk, start=start_date, end=today, auto_adjust=True, group_by='ticker', progress=False)
            if full_df is None:
                full_df = chunk_df
            else:
                full_df = pd.concat([full_df, chunk_df], axis=1)
        except Exception as e:
            print(f"  Batch {chunk_num} failed: {e}")
        
        if chunk_start + DOWNLOAD_CHUNK < len(escaped_tickers):
            time.sleep(2)

    # Metadata check - Fetch if missing entirely or missing the 'name' field
    missing_metadata = [t for t in all_tickers if t not in TICKER_INFO_DICT or "name" not in TICKER_INFO_DICT[t].get("info", {})]
    if missing_metadata:
        load_ticker_info_batch(missing_metadata, TICKER_INFO_DICT)
        write_ticker_info_file(TICKER_INFO_DICT)

    for idx, security in enumerate(securities):
        ticker = security["ticker"]
        escaped = escape_ticker(ticker)
        
        if full_df is not None and escaped in full_df.columns.levels[0]:
            ticker_df = full_df[escaped].dropna(subset=['Close'])
            if ticker_df.empty:
                ticker_data = get_yf_data(security, start_date, today)
            else:
                ticker_data = _process_yf_df(ticker_df, security)
        else:
            ticker_data = get_yf_data(security, start_date, today)
        
        # Inject metadata if available
        if ticker in TICKER_INFO_DICT:
            ticker_data["industry"] = TICKER_INFO_DICT[ticker]["info"]["industry"]
            ticker_data["sector"] = TICKER_INFO_DICT[ticker]["info"]["sector"]
            ticker_data["name"] = TICKER_INFO_DICT[ticker]["info"].get("name", ticker)
            ticker_data["marketCap"] = TICKER_INFO_DICT[ticker]["info"].get("marketCap", 0)

        tickers_dict[ticker] = ticker_data
        
        if (idx + 1) % 50 == 0 or idx == len(securities) - 1:
            elapsed = time.time() - start
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(securities) - (idx + 1))
            print(f"Processed {idx+1}/{len(securities)}. Elapsed: {int(elapsed)}s. Est. remaining: {int(remaining)}s")

    write_price_history_file(tickers_dict)

def save_data(source, securities, api_key, info = {}):
    if source == "YAHOO":
        load_prices_from_yahoo(securities, info)
    elif source == "TD_AMERITRADE":
        load_prices_from_tda(securities, api_key, info)


def main(forceTDA_legacy=None, api_key_legacy=None, full_scan_legacy=None):
    parser = argparse.ArgumentParser(description='RS Data Fetcher')
    parser.add_argument('--full', action='store_true', help='Scan all listed stocks')
    parser.add_argument('--test', action='store_true', help='Scan test set (NQ100)')
    parser.add_argument('--tda', action='store_true', help='Force TD Ameritrade source')
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 and not (isinstance(forceTDA_legacy, bool) or forceTDA_legacy is not None) else [])

    # Support legacy positional arguments
    forceTDA = args.tda if not isinstance(forceTDA_legacy, bool) else forceTDA_legacy
    api_key = API_KEY if api_key_legacy is None else api_key_legacy
    dataSource = DATA_SOURCE if not forceTDA else "TD_AMERITRADE"
    
    # Determine scan scope
    full_scan = full_scan_legacy
    if args.full: full_scan = True
    if args.test: full_scan = False

    # Load manual tips and add them to securities
    tips = load_manual_tips()
    
    # Flatten grouped tips for processing
    flattened_tips = []
    for group in tips:
        g_tickers = [t.strip().upper() for t in str(group.get("tickers", "")).split(",") if t.strip()]
        for ticker in g_tickers:
            flattened_tips.append({
                "ticker": ticker,
                "source": group.get("source", "Manual Tip"),
                "label": group.get("label", "Manual Tip"),
                "date": group.get("date", "")
            })
    tips = flattened_tips

    all_securities = list(get_resolved_securities(full_scan).values())

    # Add Market Pulse tickers
    for pt in PULSE_TICKERS:
        if not any(s["ticker"] == pt for s in all_securities):
            all_securities.append({
                "ticker": pt,
                "sector": "Market Pulse",
                "industry": "Index/ETF",
                "universe": "Market Pulse"
            })
    
    # Add tips to resolved securities list
    for tip in tips:
        ticker = tip["ticker"]
        # Find if ticker already exists in resolutions
        existing = next((s for s in all_securities if s["ticker"] == ticker), None)
        if existing:
            # Enrich existing security with tip metadata
            existing["source"] = tip.get("source", "Manual Tip")
            existing["label"] = tip.get("label", "")
            existing["date"] = tip.get("date", "")
        else:
            # Create new security entry for tip
            all_securities.append({
                "ticker": ticker,
                "sector": UNKNOWN,
                "industry": UNKNOWN,
                "universe": "Manual Tip",
                "source": tip.get("source", "Manual Tip"),
                "label": tip.get("label", ""),
                "date": tip.get("date", "")
            })
    
    save_data(dataSource, all_securities, API_KEY, {"forceTDA": forceTDA})
    write_ticker_info_file(TICKER_INFO_DICT)

if __name__ == "__main__":
    main()
