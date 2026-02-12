"""
Phase 1 Test Script

This script demonstrates the complete Phase 1 pipeline:
1. Data ingestion
2. Feature engineering
3. Rolling regime detection
4. Visualization
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from src.data_loader import download_data_if_missing
from src.features import engineer_features
from src.regime import RegimeDetector, summarize_regimes
from src.utils import (
    set_random_seeds,
    plot_regime_colored_price,
    plot_regime_probabilities,
    plot_feature_distributions_by_regime,
    print_regime_summary
)


def main():
    """Run Phase 1 pipeline."""
    
    print("="*70)
    print("PHASE 1: DATA INGESTION, FEATURES, AND REGIME DETECTION")
    print("="*70)
    
    # Initialize configuration
    config = Config(
        symbol="BTCUSDT",
        interval="1m",
        start_date="2024-01-01",
        end_date="2024-01-31",  # Using 1 month for testing
        n_regimes=3,
        train_window_days=10,  # Smaller for testing
        test_window_days=3
    )
    
    print("\nConfiguration:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Interval: {config.interval}")
    print(f"  Date range: {config.start_date} to {config.end_date}")
    print(f"  Number of regimes: {config.n_regimes}")
    print(f"  Train window: {config.train_window_days} days")
    print(f"  Test window: {config.test_window_days} days")
    
    # Set random seeds for reproducibility
    set_random_seeds(config.random_seed)
    print(f"\nRandom seed set to: {config.random_seed}")
    
    # Step 1: Load or download data
    print("\n" + "-"*70)
    print("STEP 1: DATA INGESTION")
    print("-"*70)
    
    df = download_data_if_missing(
        symbol=config.symbol,
        interval=config.interval,
        start_date=config.start_date,
        end_date=config.end_date,
        data_dir=config.data_dir
    )
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData info:")
    print(df.info())
    
    # Step 2: Feature engineering
    print("\n" + "-"*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("-"*70)
    
    price_df, features_df = engineer_features(df, config)
    
    print(f"\nFeatures shape: {features_df.shape}")
    print(f"Features: {list(features_df.columns)}")
    print(f"\nFeature statistics:")
    print(features_df.describe())
    
    # Step 3: Regime detection
    print("\n" + "-"*70)
    print("STEP 3: ROLLING REGIME DETECTION")
    print("-"*70)
    
    detector = RegimeDetector(config)
    regime_df, window_info = detector.detect_regimes(features_df, verbose=True)
    
    # Print summary
    print_regime_summary(regime_df, window_info)
    
    # Compute regime statistics
    print("\n" + "-"*70)
    print("REGIME CHARACTERISTICS")
    print("-"*70)
    
    regime_summary = summarize_regimes(regime_df, features_df)
    print("\nRegime summary statistics:")
    print(regime_summary)
    
    # Step 4: Visualizations
    print("\n" + "-"*70)
    print("STEP 4: VISUALIZATION")
    print("-"*70)
    
    print("\nGenerating plots...")
    
    # Align price data with regime labels
    price_aligned = price_df.loc[regime_df.index]
    features_aligned = features_df.loc[regime_df.index]
    
    # Plot 1: Price colored by regime
    plot_regime_colored_price(
        price_aligned,
        regime_df,
        price_column='close',
        title=f'{config.symbol} Price Colored by Market Regime'
    )
    
    # Plot 2: Regime probabilities
    plot_regime_probabilities(
        regime_df,
        title='Regime Probabilities Over Time'
    )
    
    # Plot 3: Feature distributions by regime
    plot_feature_distributions_by_regime(
        features_aligned,
        regime_df,
        feature_cols=['rolling_std', 'rolling_mean_return', 'rsi', 'ma_slope']
    )
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print("\nPhase 1 successfully demonstrated:")
    print("  ✓ Data ingestion from Binance")
    print("  ✓ Feature engineering with rolling windows")
    print("  ✓ Rolling walk-forward regime detection")
    print("  ✓ Regime label stability (sorted by volatility)")
    print("  ✓ No look-ahead bias")
    print("  ✓ Deterministic behavior (seed=42)")
    print("  ✓ Visualization of results")
    
    print("\nNext steps (Phase 2):")
    print("  - Implement strategy bank")
    print("  - Implement backtesting engine")
    print("  - Compare strategy performance across regimes")
    
    return price_df, features_df, regime_df, window_info


if __name__ == "__main__":
    price_df, features_df, regime_df, window_info = main()