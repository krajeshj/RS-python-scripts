import yfinance as yf
import json
import os
from datetime import date, timedelta

def fetch_small():
    tickers = ["SPY", "QQQ", "DIA", "IWM", "AMAT", "ASML", "GOOGL", "ROST", "GOOG"]
    today = date.today()
    data = yf.download(tickers, start=today - timedelta(days=10), end=today + timedelta(days=1))
    print(data.tail())

fetch_small()
