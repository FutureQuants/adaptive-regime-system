import pandas as pd
import numpy as np
from typing import Tuple


def compute_log_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    return np.log(prices / prices.shift(period))


def compute_rolling_mean_return(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).mean()


def compute_rolling_std(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).std()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    
    return atr


def compute_ma_slope(prices: pd.Series, window: int) -> pd.Series:

    ma = prices.rolling(window=window, min_periods=window).mean()
    ma_slope = (ma - ma.shift(window)) / ma.shift(window)
    
    return ma_slope


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:

    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_volume_zscore(volume: pd.Series, window: int) -> pd.Series:
 
    mean = volume.rolling(window=window, min_periods=window).mean()
    std = volume.rolling(window=window, min_periods=window).std()
    
    zscore = (volume - mean) / (std + 1e-8)
    
    return zscore


def engineer_features(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
  
    print("Engineering features...")
    
    features = pd.DataFrame(index=df.index)
    
    # Returns
    log_returns = compute_log_returns(df['close'], config.log_return_period)
    features['log_return'] = log_returns
    features['rolling_mean_return'] = compute_rolling_mean_return(
        log_returns, 
        config.rolling_mean_return_window
    )
    
    # Volatility
    features['rolling_std'] = compute_rolling_std(
        log_returns,
        config.rolling_std_window
    )
    features['atr'] = compute_atr(
        df['high'],
        df['low'],
        df['close'],
        config.atr_window
    )
    
    # Trend / Momentum
    features['ma_slope'] = compute_ma_slope(
        df['close'],
        config.ma_slope_window
    )
    features['rsi'] = compute_rsi(
        df['close'],
        config.rsi_window
    )
    
    # Volume
    features['rolling_mean_volume'] = df['volume'].rolling(
        window=config.rolling_mean_volume_window,
        min_periods=config.rolling_mean_volume_window
    ).mean()
    features['volume_zscore'] = compute_volume_zscore(
        df['volume'],
        config.volume_zscore_window
    )
    
    # Drop rows with NaN values
    initial_count = len(features)
    features_clean = features.dropna()
    dropped_count = initial_count - len(features_clean)
    
    print(f"Dropped {dropped_count} initial rows with NaN values")
    print(f"Remaining bars: {len(features_clean)}")
    
    # Align price data with features
    price_df = df.loc[features_clean.index].copy()
    
    # Validate alignment
    assert len(price_df) == len(features_clean), "Price and features not aligned"
    assert (price_df.index == features_clean.index).all(), "Indices don't match"
    
    print(f"Features shape: {features_clean.shape}")
    print(f"Feature columns: {list(features_clean.columns)}")
    
    return price_df, features_clean