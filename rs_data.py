#!/usr/bin/env python
import requests
import json
import time
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

def get_securities(url, ticker_pos = 1, table_pos = 1, sector_offset = 1, industry_offset = 1, universe = "N/A"):
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.findAll('table', {'class': 'wikitable sortable'})[table_pos-1]
    secs = {}
    for row in table.findAll('tr')[table_pos:]:
        sec = {}
        sec["ticker"] = row.findAll('td')[ticker_pos-1].text.strip()
        sec["sector"] = row.findAll('td')[ticker_pos-1+sector_offset].text.strip()
        sec["industry"] = row.findAll('td')[ticker_pos-1+sector_offset+industry_offset].text.strip()
        sec["universe"] = universe
        secs[sec["ticker"]] = sec
    with open(os.path.join(DIR, "tmp", "tickers.pickle"), "wb") as f:
        pickle.dump(secs, f)
    return secs


def get_resolved_securities():
    tickers = {REFERENCE_TICKER: REF_TICKER}
    if ALL_STOCKS:
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
        tickers.update(get_securities('https://en.wikipedia.org/wiki/Nasdaq-100', 2, 3, universe="Nasdaq 100"))
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

    for entry in results:
        sec = {}
        values = entry.split('|')
        ticker = values[ticker_column]
        if re.match(r'^[A-Z]+$', ticker) and values[etf_column] == "N" and values[test_column] == "N":
            sec["ticker"] = ticker
            sec["sector"] = UNKNOWN
            sec["industry"] = UNKNOWN
            sec["universe"] = exchange_from_symbol(values[exchange_column])
            tickers[sec["ticker"]] = sec

    return tickers

SECURITIES = get_resolved_securities().values()

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
    ticker_response["sector"] = security["sector"]
    ticker_response["industry"] = security["industry"]
    ticker_response["universe"] = security["universe"]
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
def load_ticker_info_batch(tickers, info_dict):
    """Fetches ticker info in parallel."""
    print(f"Fetching metadata for {len(tickers)} tickers in parallel...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(load_ticker_info, t, {}): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                res = future.result()
                if res:
                    info_dict.update(res)
            except Exception as e:
                print(f"Error fetching metadata for {ticker}: {e}")

def load_ticker_info(ticker, info_dict_ignored):
    """Fetches info for a single ticker. Returns a dict to be merged."""
    escaped_ticker = escape_ticker(ticker)
    try: 
        info = yf.Ticker(escaped_ticker).info
        if not info:
            return None
        return {
            ticker: {
                "info": {
                    "industry": get_info_from_dict(info, "industry"),
                    "sector": get_info_from_dict(info, "sector"),
                    "marketCap": info.get("marketCap", 0)
                }
            }
        }
    except Exception as e:
        # print(f"Error fetching metadata for {escaped_ticker}: {e}")
        return None

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


def get_market_cap(ticker):
    """
    Fetches market cap for a ticker. 
    In a high-performance environment, it's better to fetch these in bulk
    or use a more efficient API than yahoo_fin for large universes.
    """
    try:
        # Ticker info often contains market cap, check cache first if we ever add it there
        quote = yf.Ticker(ticker).info
        return quote.get("marketCap", 1000000001)
    except Exception:
        try:
            quote_table = get_quote_table(ticker)
            market_cap = quote_table.get("Market Cap", "0")
            return convert_string_to_numeric(market_cap)
        except Exception as e:
            print(f"Error fetching Market Cap for {ticker}: {e}")
            return 1000000001





 







   
   


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
    
    # Batch download
    print(f"Downloading data for {len(escaped_tickers)} tickers...")
    try:
        full_df = yf.download(escaped_tickers, start=start_date, end=today, auto_adjust=True, group_by='ticker', progress=True)
    except Exception as e:
        print(f"Batch download failed: {e}. Falling back to sequential.")
        full_df = None

    # Metadata check
    missing_metadata = [t for t in all_tickers if t not in TICKER_INFO_DICT]
    if missing_metadata:
        load_ticker_info_batch(missing_metadata, TICKER_INFO_DICT)
        write_ticker_info_file(TICKER_INFO_DICT)

    for idx, security in enumerate(securities):
        ticker = security["ticker"]
        escaped = escape_ticker(ticker)
        
        if full_df is not None and escaped in full_df.columns.levels[0]:
            ticker_df = full_df[escaped].dropna(subset=['Close'])
            ticker_data = _process_yf_df(ticker_df, security)
        else:
            ticker_data = get_yf_data(security, start_date, today)
        
        # Inject metadata if available
        if ticker in TICKER_INFO_DICT:
            ticker_data["industry"] = TICKER_INFO_DICT[ticker]["info"]["industry"]
            ticker_data["sector"] = TICKER_INFO_DICT[ticker]["info"]["sector"]
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


def main(forceTDA = False, api_key = API_KEY):
    dataSource = DATA_SOURCE if not forceTDA else "TD_AMERITRADE"
    
    # Load manual tips and add them to securities
    tips = load_manual_tips()
    all_securities = list(SECURITIES)
    
    # Track which tickers are manual tips for later info fetching
    manual_tickers = []
    for tip in tips:
        ticker = tip["ticker"]
        # Create security object if not already present
        if not any(s["ticker"] == ticker for s in all_securities):
            all_securities.append({
                "ticker": ticker,
                "sector": UNKNOWN,
                "industry": UNKNOWN,
                "universe": "Manual Tip",
                "source": tip.get("source", "Manual Tip")
            })
            manual_tickers.append(ticker)
        else:
            # Update source for existing security if it's in tips
            for s in all_securities:
                if s["ticker"] == ticker:
                    s["source"] = tip.get("source", "Manual Tip")
                    break

    save_data(dataSource, all_securities, api_key, {"forceTDA": forceTDA})
    write_ticker_info_file(TICKER_INFO_DICT)

if __name__ == "__main__":
    main()
