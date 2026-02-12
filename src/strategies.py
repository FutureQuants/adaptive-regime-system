import pandas as pd
import numpy as np
from typing import Dict, Tuple


class Strategy:
    # The base class for trading strategies
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        config
    ) -> Tuple[pd.Series, pd.Series]:
  
        raise NotImplementedError


class TrendFollowingStrategy(Strategy):
    
    def __init__(self):
        super().__init__("Trend Following")
    
    def generate_signals(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        config
    ) -> Tuple[pd.Series, pd.Series]:
  
        close = price_df['close']
        
        # Calculate moving averages
        fast_ma = close.rolling(window=config.trend_fast_ma, min_periods=config.trend_fast_ma).mean()
        slow_ma = close.rolling(window=config.trend_slow_ma, min_periods=config.trend_slow_ma).mean()
        
        # Generate position: 1 when fast > slow, 0 otherwise
        position = (fast_ma > slow_ma).astype(int)
        
        # Generate entry/exit signals
        signal = position.diff()
        
        return position, signal


class MeanReversionStrategy(Strategy):

    def __init__(self):
        super().__init__("Mean Reversion")
    
    def generate_signals(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        config
    ) -> Tuple[pd.Series, pd.Series]:

        rsi = features_df['rsi']
        
        # Initialize position series
        position = pd.Series(0, index=rsi.index, dtype=int)
        
        # Track current position state
        in_position = False
        
        for i in range(len(rsi)):
            if pd.isna(rsi.iloc[i]):
                position.iloc[i] = 0
                continue
            
            if not in_position:
                # Check for entry: RSI < 30
                if rsi.iloc[i] < config.mean_reversion_rsi_entry:
                    in_position = True
                    position.iloc[i] = 1
                else:
                    position.iloc[i] = 0
            else:
                # Already in position, check for exit: RSI > 55
                if rsi.iloc[i] > config.mean_reversion_rsi_exit:
                    in_position = False
                    position.iloc[i] = 0
                else:
                    position.iloc[i] = 1
        
        # Generate entry/exit signals
        signal = position.diff()
        
        return position, signal


class VolatilityBreakoutStrategy(Strategy):
    
    def __init__(self):
        super().__init__("Volatility Breakout")
    
    def generate_signals(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        config
    ) -> Tuple[pd.Series, pd.Series]:

        close = price_df['close']
        
        # Calculate rolling highs and lows
        entry_threshold = close.shift(1).rolling(
            window=config.breakout_entry_window,
            min_periods=config.breakout_entry_window
        ).max()
        
        exit_threshold = close.shift(1).rolling(
            window=config.breakout_exit_window,
            min_periods=config.breakout_exit_window
        ).min()
        
        # Initialize position series
        position = pd.Series(0, index=close.index, dtype=int)
        
        # Track current position state
        in_position = False
        
        for i in range(len(close)):
            if pd.isna(entry_threshold.iloc[i]) or pd.isna(exit_threshold.iloc[i]):
                position.iloc[i] = 0
                continue
            
            if not in_position:
                # Check for entry: close > 20-bar high
                if close.iloc[i] > entry_threshold.iloc[i]:
                    in_position = True
                    position.iloc[i] = 1
                else:
                    position.iloc[i] = 0
            else:
                # Already in position, check for exit: close < 10-bar low
                if close.iloc[i] < exit_threshold.iloc[i]:
                    in_position = False
                    position.iloc[i] = 0
                else:
                    position.iloc[i] = 1
        
        # Generate entry/exit signals
        signal = position.diff()
        
        return position, signal


class FlatStrategy(Strategy):

    def __init__(self):
        super().__init__("Flat")
    
    def generate_signals(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        config
    ) -> Tuple[pd.Series, pd.Series]:
        # Generate flat signals (always 0).
        position = pd.Series(0, index=price_df.index, dtype=int)
        signal = pd.Series(0, index=price_df.index, dtype=int)
        
        return position, signal


def generate_all_strategies(
    price_df: pd.DataFrame,
    features_df: pd.DataFrame,
    config
) -> Dict[str, pd.Series]:

    strategies = {
        'trend': TrendFollowingStrategy(),
        'mean_reversion': MeanReversionStrategy(),
        'breakout': VolatilityBreakoutStrategy(),
        'flat': FlatStrategy()
    }
    
    positions = {}
    
    for key, strategy in strategies.items():
        print(f"Generating signals for {strategy.name}...")
        position, signal = strategy.generate_signals(price_df, features_df, config)
        positions[key] = position
    
    return positions


def get_strategy_by_index(index: int) -> str:

    mapping = {
        0: 'trend',
        1: 'mean_reversion',
        2: 'breakout',
        3: 'flat'
    }
    return mapping.get(index, 'flat')


def get_strategy_index(name: str) -> int:

    mapping = {
        'trend': 0,
        'mean_reversion': 1,
        'breakout': 2,
        'flat': 3
    }
    return mapping.get(name, 3)