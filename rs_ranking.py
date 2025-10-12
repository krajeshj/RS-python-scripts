import sys
import pandas as pd
import numpy as np
import json
import os
from datetime import date
from scipy.stats import linregress
import yaml
from rs_data import TD_API, cfg, read_json
from functools import reduce
import csv

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
TICKER_INFO_DICT = read_json(TICKER_INFO_FILE)

TITLE_RANK = "Rank"
TITLE_TICKER = "Ticker"
TITLE_TICKERS ="Tickers"
TITLE_MINERVINI="Minervini"
TITLE_SECTOR = "Sector"
TITLE_INDUSTRY = "Industry"
TITLE_UNIVERSE = "Universe" if not ALL_STOCKS else "Exchange"
TITLE_PERCENTILE = "Percentile"
TITLE_1M = "1 Month Ago"
TITLE_3M = "3 Months Ago"
TITLE_6M = "6 Months Ago"
TITLE_RS = "Relative Strength"
TITLE_RMV = "RMV"

if not os.path.exists('output'):
    os.makedirs('output')

def read_json(json_file):
    with open(json_file, "r") as fp:
        return json.load(fp)

def calculate_rmv(closes: pd.Series, highs: pd.Series = None, lows: pd.Series = None, lookback_period: int = 15):
    """
    Calculate Relative Market Volatility (RMV) over a lookback period.
    
    Based on DeepVue.com methodology:
    - Uses highs and lows over the specified period
    - Calculates volatility based on price range (high-low) relative to price level
    - Returns a score between 0-100 where 0 = tight price action, 100 = high volatility
    
    Args:
        closes: Pandas Series of closing prices
        highs: Pandas Series of high prices (optional, will use closes if not provided)
        lows: Pandas Series of low prices (optional, will use closes if not provided)
        lookback_period: Number of days to look back (default: 15)
    
    Returns:
        RMV value (float) between 0-100
    """
    try:
        if len(closes) < lookback_period:
            return 50.0  # Default to middle value if insufficient data
        
        # Use closes as fallback for highs/lows if not provided
        if highs is None:
            highs = closes
        if lows is None:
            lows = closes
        
        # Ensure all series have the same length
        min_length = min(len(closes), len(highs), len(lows))
        closes = closes.tail(min_length)
        highs = highs.tail(min_length)
        lows = lows.tail(min_length)
        
        if len(closes) < lookback_period:
            return 50.0
        
        # Calculate daily price ranges (high - low)
        daily_ranges = highs - lows
        
        # Calculate average price over the period
        avg_price = closes.tail(lookback_period).mean()
        
        # Calculate average daily range over the lookback period
        avg_daily_range = daily_ranges.tail(lookback_period).mean()
        
        # Avoid division by zero
        if avg_price == 0 or pd.isna(avg_price) or pd.isna(avg_daily_range):
            return 50.0
        
        # Calculate volatility as percentage of average price
        volatility_pct = (avg_daily_range / avg_price) * 100
        
        # Scale to 0-100 range (this is an approximation - DeepVue uses proprietary scaling)
        # Based on analysis of ORCL data: volatility_pct = 4.15%, target RMV = 27.30
        # Required scaling factor = 27.30 / 4.15 = 6.58
        # Using slightly higher factor to account for different methodologies
        rmv = min(100.0, max(0.0, volatility_pct * 7.0))
        
        # Handle NaN or infinite values
        if pd.isna(rmv) or np.isinf(rmv):
            return 50.0
        
        return round(rmv, 2)
        
    except Exception as e:
        print(f"Error calculating RMV: {e}")
        return 50.0

def relative_strength_orig(closes: pd.Series, closes_ref: pd.Series):
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)
    rs = (rs_stock/rs_ref - 1) * 100
    rs = int(rs*100) / 100 # round to 2 decimals
    return rs

def relative_strength(closes: pd.Series, closes_ref: pd.Series):
    try:
    
        rs1 = quarters_rs(closes,closes_ref,1)
        rs2 = quarters_rs(closes,closes_ref,2)
        rs3 = quarters_rs(closes,closes_ref,3)
        rs4 = quarters_rs(closes,closes_ref,4)
        rs = 0.4*rs1 + 0.2*rs2 + 0.2*rs3 + 0.2*rs4
        #rs = int(rs*10000) / 100 # round to 2 decimals
        return rs
    except:
        print("Exception in relative strength - rs = 0")
        return 0.00000000000001


def strength(closes: pd.Series):
    """Calculates the performance of the last year (most recent quarter is weighted double)"""
    try:
        quarters1 = quarters_perf(closes, 1)
        quarters2 = quarters_perf(closes, 2)
        quarters3 = quarters_perf(closes, 3)
        quarters4 = quarters_perf(closes, 4)
        return 0.4*quarters1 + 0.2*quarters2 + 0.2*quarters3 + 0.2*quarters4
    except:
        return 0

def quarters_perf(closes: pd.Series, n):
    length = min(len(closes), n*int(252/4))
    prices = closes.tail(length)
    pct_chg = prices.pct_change().dropna()
    print(" rs_data pct_cjhange is \n",pct_chg)
    perf_cum = (pct_chg + 1).cumprod() - 1
    print("quarter_perf", perf_cum)
    return perf_cum.tail(1).item()

def quarters_rs(closes: pd.Series, closes_ref: pd.Series, n):
    try: 
        length = min(len(closes), n*int(252/4))
        df_prices_n = closes.tail(length).dropna()
        prices_n = df_prices_n.head(1).item()

        df_prices_ref_n = closes_ref.tail(length).dropna()
        prices_ref_n = df_prices_ref_n.head(1).item()


        prices = closes.tail(1).item()
        prices_ref = closes_ref.tail(1).item()

        #print("rs_data quarters_rs prices     :", prices)
        #print("rs_data quarters_rs prices_ref :", prices_ref)

    
    
        rs_n = (prices / prices_n) / (prices_ref/prices_ref_n)

        final_rs_n = rs_n

        #return final_rs_n.tail(1).item()
        #print("rs_data quarters_rs return value : ", final_rs_n)

        
        return final_rs_n
    except:
        return 0

    #pct_chg = prices.pct_change().dropna()
    #erf_cum = (pct_chg + 1).cumprod() - 1
    #return perf_cum.tail(1).item()




def rankings(test_mode=False, test_tickers=None):
    """Returns a dataframe with percentile rankings for relative strength"""
    json = read_json(PRICE_DATA)
    relative_strengths = []
    ranks = []
    industries = {}
    ind_ranks = []
    stock_rs = {}
    ref = json[REFERENCE_TICKER]
    
    # Test mode: only process specific tickers
    if test_mode and test_tickers:
        tickers_to_process = [ticker for ticker in test_tickers if ticker in json]
        print(f"Test mode: Processing {len(tickers_to_process)} tickers: {tickers_to_process}")
    else:
        tickers_to_process = json.keys()
    
    for ticker in tickers_to_process:
        try :
            if  ((json[ticker]["skip_calc"] == 0)):
                #print(" Starting calculation for  ticker - ", ticker)  
                if not cfg("SP500") and json[ticker]["universe"] == "S&P 500":
                    continue
                if not cfg("SP400") and json[ticker]["universe"] == "S&P 400":
                    continue
                if not cfg("SP600") and json[ticker]["universe"] == "S&P 600":
                    continue
                if not cfg("NQ100") and json[ticker]["universe"] == "Nasdaq 100":
                    continue
                #try:
                #closes = list(map(lambda candle: candle["close"], json[ticker]["candles"]))
                #print("This is calculation for  ticker - ", ticker)                

                closes = list(map(lambda candle: candle["close"], json[ticker]["candles"]))
                highs = list(map(lambda candle: candle["high"], json[ticker]["candles"]))
                lows = list(map(lambda candle: candle["low"], json[ticker]["candles"]))

                closes_ref = list(map(lambda candle: candle["close"], ref["candles"]))
                industry = TICKER_INFO_DICT[ticker]["info"]["industry"] if json[ticker]["industry"] == "unknown" else json[ticker]["industry"]
                sector = TICKER_INFO_DICT[ticker]["info"]["sector"] if json[ticker]["sector"] == "unknown" else json[ticker]["sector"]
                if len(closes) >= 6*20 and industry != "n/a" and len(industry.strip()) > 0:
                    closes_series = pd.Series(closes)
                    highs_series = pd.Series(highs)
                    lows_series = pd.Series(lows)
                    closes_ref_series = pd.Series(closes_ref)
                    rs = relative_strength(closes_series, closes_ref_series)
                    rmv = calculate_rmv(closes_series, highs_series, lows_series)
                    month = 20
                    tmp_percentile = 100
                    rs1m = relative_strength(closes_series.head(-1*month), closes_ref_series.head(-1*month))
                    rs3m = relative_strength(closes_series.head(-3*month), closes_ref_series.head(-3*month))
                    rs6m = relative_strength(closes_series.head(-6*month), closes_ref_series.head(-6*month))
                                    # if rs is too big assume there is faulty price data
                    #print(f'Ticker {ticker} has {rs}.')
                    #if rs < 8000:
                    if (( closes_series.iloc[-1] > 12)):

                        # stocks output
                        ranks.append(len(ranks)+1)
                        #relative_strengths.append((0, ticker, sector, industry, json[ticker]["universe"], rs, tmp_percentile, rs1m, rs3m, rs6m))
                        relative_strengths.append((0, ticker, json[ticker]["minervini"], sector, industry, json[ticker]["universe"], rs, tmp_percentile, rs1m, rs3m, rs6m, rmv))                     
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
                            ind_ranks.append(len(ind_ranks)+1)
                        industries[industry][TITLE_RS].append(rs)
                        industries[industry][TITLE_1M].append(rs1m)
                        industries[industry][TITLE_3M].append(rs3m)
                        industries[industry][TITLE_6M].append(rs6m)
                        industries[industry][TITLE_TICKERS].append(ticker)
                        #print(f'Ticker {ticker} included')



                        #ranks.append(len(ranks)+1)
                        #relative_strengths.append((0, ticker, json[ticker]["sector"], json[ticker]["universe"], rs, tmp_percentile, rs1m, rs3m, rs6m))
            else:
                #print(f'Ticker {ticker} filtered out')
                continue
        except:
            print(f'Ticker {ticker} has corrupted data.')
            continue
            
                      
    dfs = []
    dfs_mnrvni = []
    suffix = ''
    df = pd.DataFrame(relative_strengths, columns=[TITLE_RANK, TITLE_TICKER, TITLE_MINERVINI, TITLE_SECTOR, TITLE_INDUSTRY, TITLE_UNIVERSE, TITLE_RS, TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M, TITLE_RMV])
 
    df[TITLE_PERCENTILE] = pd.qcut(df[TITLE_RS], 100, precision=64, labels=False,duplicates='drop' )
    df[TITLE_1M] = pd.qcut(df[TITLE_1M], 100, precision=64, labels=False,duplicates='drop')
    df[TITLE_3M] = pd.qcut(df[TITLE_3M], 100, precision=64,labels=False,duplicates='drop')
    df[TITLE_6M] = pd.qcut(df[TITLE_6M], 100, precision=64, labels=False,duplicates='drop')
    df = df.sort_values(([TITLE_RS]), ascending=False)
    df[TITLE_RANK] = ranks
    out_tickers_count = 0
    for index, row in df.iterrows():
        if ((row[TITLE_PERCENTILE] >= MIN_PERCENTILE) ):
            out_tickers_count = out_tickers_count + 1
    df = df.head(out_tickers_count)
    # drop rows which don't meet Minervini criteria
    # drop rows which have 6m/3m RS ranking less than 25
    dfm = df[df['Minervini'] > 6  ]
    print(df.head())

    #dfm1 = df[df['Minervini'] > 6  ]
    #dfm2 = dfm1[ dfm1['6 Months Ago'] > 25 ]
    #dfm = dfm2[ dfm2['3 Months Ago'] > 25 ]
    
    
    
    #sort dfm 7-8 Then by Rank 
    dfm = dfm.sort_values(([TITLE_MINERVINI,  TITLE_RANK]), ascending=[False,True])
    
    
    print(df.head())
 
    print(dfm.head())
 


    df.to_csv(os.path.join(DIR, "output", f'rs_stocks{suffix}.csv'), index = False)
    dfm.to_csv(os.path.join(DIR, "output", f'rs_stocks_minervini.csv'), index = False)
    
    # Create low RMV list (RMV <= 15, sorted by RMV ascending)
    df_low_rmv = df[df[TITLE_RMV] <= 15].copy()
    if not df_low_rmv.empty:
        df_low_rmv = df_low_rmv.sort_values(TITLE_RMV, ascending=True)
        df_low_rmv[TITLE_RANK] = range(1, len(df_low_rmv) + 1)  # Re-rank by RMV
        df_low_rmv.to_csv(os.path.join(DIR, "output", "rs_stocks_low_rmv.csv"), index=False)
        print(f"\nLow RMV stocks (RMV <= 15): {len(df_low_rmv)} stocks")
        print(df_low_rmv[[TITLE_RANK, TITLE_TICKER, TITLE_RMV, TITLE_RS, TITLE_SECTOR, TITLE_INDUSTRY]].head(10))
    else:
        print("\nNo stocks found with RMV <= 15")
    
    # Also create a list showing all stocks sorted by RMV (for reference)
    df_sorted_rmv = df.copy()
    df_sorted_rmv = df_sorted_rmv.sort_values(TITLE_RMV, ascending=True)
    df_sorted_rmv[TITLE_RANK] = range(1, len(df_sorted_rmv) + 1)  # Re-rank by RMV
    df_sorted_rmv.to_csv(os.path.join(DIR, "output", "rs_stocks_by_rmv.csv"), index=False)
    print(f"\nAll stocks sorted by RMV (lowest first): {len(df_sorted_rmv)} stocks")
    print(df_sorted_rmv[[TITLE_RANK, TITLE_TICKER, TITLE_RMV, TITLE_RS, TITLE_SECTOR, TITLE_INDUSTRY]].head(10))
    
    # Create rmv_rs.csv with all requested columns, filtered for RMV <= 15
    df_rmv_rs = df[df[TITLE_RMV] <= 15].copy()
    if not df_rmv_rs.empty:
        df_rmv_rs = df_rmv_rs.sort_values(TITLE_RMV, ascending=True)
        # Select columns: RMV, Ticker, Minervini, Sector, Industry, Percentile, Relative Strength, 1 Month Ago
        df_rmv_rs_output = df_rmv_rs[[TITLE_RMV, TITLE_TICKER, TITLE_MINERVINI, TITLE_SECTOR, TITLE_INDUSTRY, TITLE_PERCENTILE, TITLE_RS, TITLE_1M]].copy()
        df_rmv_rs_output.to_csv(os.path.join(DIR, "output", "rmv_rs.csv"), index=False)
        print(f"\nRMV-RS list (RMV <= 15): {len(df_rmv_rs_output)} stocks")
        print(df_rmv_rs_output.head(10))
    else:
        print("\nNo stocks found with RMV <= 15 for rmv_rs.csv")
    try:
        dfs.append(df)
        dfs_mnrvni.append(dfm)
        print(f'Ticker {ticker} data has been added.')
    except:
        print(f'Ticker {ticker} coukd not be added.')

    
    try: 
        list_of_dfm_tickers = dfm[TITLE_TICKER].to_list()
        #print(f"'{list_of_dfm_tickers}'")

        list_of_dfm_tickers = ", ".join(map(str,list_of_dfm_tickers))

        print(f"'{list_of_dfm_tickers}'")
    
        with open(os.path.join(DIR, "output", f'Minervini_list.csv'), 'w' )as f:
            write = csv.writer(f)
            f.writelines(list_of_dfm_tickers)
    except:
        print(f'Minervini_list not created')
    
    print("Heartbeat 1 \n")
# industries
    def getDfView(industry_entry):
        return industry_entry["info"]
    def sum(a,b):
        return a+b
    def getRsAverage(industries, industry, column):
        rs = reduce(sum, industries[industry][column])/len(industries[industry][column])
        rs = int(rs*100) / 100 # round to 2 decimals
        return rs
    def rs_for_stock(ticker):
        return stock_rs[ticker]
    def getTickers(industries, industry):
        return ",".join(sorted(industries[industry][new_func()], key=rs_for_stock, reverse=True))

    def new_func():
        return TITLE_TICKERS

        

    # remove industries with only one stock
    filtered_industries = filter(lambda i: len(i[TITLE_TICKERS]) > 1, list(industries.values()))
    filtered_industries_list = list(filtered_industries)
    
    if len(filtered_industries_list) == 0:
        print("No industries with multiple stocks found. Skipping industry analysis.")
        return dfs
    
    df_industries = pd.DataFrame(map(getDfView, filtered_industries_list), columns=[TITLE_RANK, TITLE_INDUSTRY, TITLE_SECTOR, TITLE_RS,  TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M])
 
    df_industries[TITLE_RS] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_RS), axis=1)
    df_industries[TITLE_1M] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_1M), axis=1)
    df_industries[TITLE_3M] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_3M), axis=1)
    df_industries[TITLE_6M] = df_industries.apply(lambda row: getRsAverage(industries, row[TITLE_INDUSTRY], TITLE_6M), axis=1)
    
 
    #df_industries[TITLE_PERCENTILE] = df_industries[TITLE_RS].transform(lambda x: pd.qcut(x.rank(method='first'), 100, labels=False))
    #df_industries[TITLE_1M]         = df_industries[TITLE_1M].transform(lambda x: pd.qcut(x.rank(method='first'), 100, labels=False))
    #df_industries[TITLE_3M]         = df_industries[TITLE_3M].transform(lambda x: pd.qcut(x.rank(method='first'), 100, labels=False))
    #df_industries[TITLE_6M]         = df_industries[TITLE_6M].transform(lambda x: pd.qcut(x.rank(method='first'), 100, labels=False))
    #df_industries[TITLE_TICKERS]    = df_industries.apply(lambda row: getTickers(industries, row[TITLE_INDUSTRY]), axis=1)
    ##df_industries = df_industries.sort_values(([TITLE_RS]), ascending=False)
    df_industries[TITLE_PERCENTILE] = pd.qcut(df_industries[TITLE_RS], 100, labels=False, duplicates="drop")
    df_industries[TITLE_1M]         = pd.qcut(df_industries[TITLE_1M], 100, labels=False, duplicates="drop")
    df_industries[TITLE_3M]         = pd.qcut(df_industries[TITLE_3M], 100, labels=False, duplicates="drop")
    df_industries[TITLE_6M]         = pd.qcut(df_industries[TITLE_6M], 100, labels=False, duplicates="drop")
    df_industries[TITLE_TICKERS] = df_industries.apply(lambda row: getTickers(industries, row[TITLE_INDUSTRY]), axis=1)
    #df_industries = df_industries.sort_values(([TITLE_RS]), ascending=False)
    
    df_industries = df_industries.sort_values(([TITLE_PERCENTILE]), ascending=False)
 
    ind_ranks = ind_ranks[:len(df_industries)]
    df_industries[TITLE_RANK] = ind_ranks
 
    df_industries.to_csv(os.path.join(DIR, "output", f'rs_industries{suffix}.csv'), index = False)
    dfs.append(df_industries)
 
    return dfs


def main(skipEnter = False, test_mode=False, test_tickers=None):
    if test_mode:
        print("Running in TEST MODE")
        ranks = rankings(test_mode=True, test_tickers=test_tickers)
    else:
        ranks = rankings()
    
    if ranks:
        print(ranks[0])
        print("***\nYour 'rs_stocks.csv' is in the output folder.\n***")
    else:
        print("No data processed.")
    
    if not skipEnter and cfg("EXIT_WAIT_FOR_ENTER"):
        input("Press Enter key to exit...")

if __name__ == "__main__":
    main()