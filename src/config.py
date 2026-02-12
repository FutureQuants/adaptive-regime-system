from dataclasses import dataclass
from datetime import datetime


@dataclass
class Config:
    
    # Data parameters
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    start_date: str = "2024-01-01"
    end_date: str = "2024-03-31"
    
    # Feature engineering parameters
    log_return_period: int = 1
    rolling_mean_return_window: int = 60
    rolling_std_window: int = 60
    atr_window: int = 14
    ma_slope_window: int = 50
    rsi_window: int = 14
    rolling_mean_volume_window: int = 60
    volume_zscore_window: int = 60
    
    # Regime detection parameters
    n_regimes: int = 3
    gmm_covariance_type: str = "full"
    gmm_n_init: int = 1
    gmm_random_state: int = 42
    
    # Walk-forward parameters
    train_window_days: int = 60
    test_window_days: int = 14
    bars_per_day: int = 1440  # for 1-minute bars
    
    # Strategy parameters
    transaction_cost: float = 0.001  # 0.1%
    switching_penalty: float = 0.0005
    
    # Strategy-specific parameters
    trend_fast_ma: int = 20
    trend_slow_ma: int = 100
    mean_reversion_rsi_entry: int = 30
    mean_reversion_rsi_exit: int = 55
    breakout_entry_window: int = 20
    breakout_exit_window: int = 10
    
    # RL parameters
    rl_random_seed: int = 42
    rl_learning_rate: float = 0.0003
    rl_n_steps: int = 2048
    rl_batch_size: int = 64
    rl_n_epochs: int = 10
    rl_gamma: float = 0.99
    
    # System parameters
    random_seed: int = 42
    
    # Data paths
    data_dir: str = "data"
    
    def get_train_window_bars(self) -> int:
        # Calculate number of bars in training window
        return self.train_window_days * self.bars_per_day
    
    def get_test_window_bars(self) -> int:
        # Calculate number of bars in test window
        return self.test_window_days * self.bars_per_day
    
    def get_data_filename(self) -> str:
        # Generate filename for data CSV
        return f"{self.data_dir}/{self.symbol}_{self.interval}_{self.start_date}_{self.end_date}.csv"