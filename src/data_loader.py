import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import requests
import time


def download_binance_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    data_dir: str = "data"
) -> pd.DataFrame:
    
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    # Convert dates to milliseconds timestamp
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    # Binance API endpoint
    url = "https://api.binance.com/api/v3/klines"
    
    all_data = []
    current_ts = start_ts
    
    # Download in chunks (max 1000 candles per request)
    while current_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update current timestamp to last candle's close time + 1ms
            current_ts = data[-1][6] + 1
            
            # Rate limiting
            time.sleep(0.1)
            
            print(f"Downloaded {len(all_data)} bars...", end='\r')
            
        except Exception as e:
            print(f"\nError downloading data: {e}")
            break
    
    print(f"\nTotal bars downloaded: {len(all_data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Keep only OHLCV columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df[['open', 'high', 'low', 'close', 'volume']] = \
        df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Ensure strictly increasing index
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    # Validate data
    assert df.index.is_monotonic_increasing, "Index not strictly increasing"
    assert not df.index.has_duplicates, "Duplicate timestamps found"
    assert not df.isnull().any().any(), "Missing values found"
    
    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")


def load_data(filepath: str) -> pd.DataFrame:

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    # Validate data integrity
    assert df.index.is_monotonic_increasing, "Index not strictly increasing"
    assert not df.index.has_duplicates, "Duplicate timestamps found"
    
    print(f"Loaded {len(df)} bars from {filepath}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def download_data_if_missing(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    data_dir: str = "data"
) -> pd.DataFrame:

    filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Loading existing data from {filepath}")
        return load_data(filepath)
    else:
        print(f"Data file not found. Downloading...")
        df = download_binance_data(symbol, interval, start_date, end_date, data_dir)
        save_data(df, filepath)
        return df