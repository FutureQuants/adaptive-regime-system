"""
Phase 3 Test Script - RL Meta-Controller

Tests the RL implementation with synthetic data.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from src.features import engineer_features
from src.regime import RegimeDetector
from src.strategies import generate_all_strategies
from src.backtester import Backtester
from src.rl_env import StrategySelectionEnv
from src.rl_trainer import RLTrainer, SimplePolicy
from src.metrics import calculate_all_metrics, create_metrics_dataframe
from src.utils import set_random_seeds


def generate_synthetic_data(n_bars=30000, seed=42):
    """Generate synthetic OHLCV data with regime-like behavior."""
    np.random.seed(seed)
    
    start = pd.Timestamp('2024-01-01', tz='UTC')
    dates = pd.date_range(start, periods=n_bars, freq='1min')
    
    prices = np.zeros(n_bars)
    prices[0] = 10000
    
    for i in range(1, n_bars):
        if i < n_bars // 3:
            ret = np.random.normal(0.00001, 0.0002)
        elif i < 2 * n_bars // 3:
            ret = np.random.normal(0.00005, 0.0003)
        else:
            ret = np.random.normal(0, 0.0008)
        prices[i] = prices[i-1] * (1 + ret)
    
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.001, n_bars))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.001, n_bars))
    df['volume'] = np.random.uniform(100, 1000, n_bars)
    
    return df


def main():
    print("="*70)
    print("PHASE 3: RL META-CONTROLLER")
    print("="*70)
    
    config = Config(
        n_regimes=3,
        train_window_days=7,
        test_window_days=2,
        random_seed=42
    )
    
    set_random_seeds(42)
    
    print("\n1. Generating data...")
    df = generate_synthetic_data(n_bars=30000, seed=42)
    print(f"   ✓ {len(df)} bars")
    
    print("\n2. Features...")
    price_df, features_df = engineer_features(df, config)
    print(f"   ✓ {features_df.shape[1]} features")
    
    print("\n3. Regimes...")
    detector = RegimeDetector(config)
    regime_df, window_info = detector.detect_regimes(features_df, verbose=False)
    print(f"   ✓ {len(regime_df)} bars, {len(window_info)} windows")
    
    print("\n4. Strategies...")
    strategy_positions = generate_all_strategies(price_df, features_df, config)
    print(f"   ✓ {len(strategy_positions)} strategies")
    
    print("\n5. RL Environment...")
    test_regime_df = regime_df.iloc[:1000]
    
    env = StrategySelectionEnv(
        price_df, features_df, test_regime_df,
        strategy_positions, config, is_training=True
    )
    
    print(f"   ✓ State dim: {env.get_state_dim()}, Action dim: {env.get_action_dim()}")
    
    state = env.reset()
    next_state, reward, done, info = env.step(0)
    print(f"   ✓ Step test: reward={reward:.6f}")
    
    print("\n6. RL Policy...")
    policy = SimplePolicy(env.get_state_dim(), env.get_action_dim(), seed=42)
    action = policy.select_action(state, deterministic=True)
    print(f"   ✓ Action: {action}")
    
    print("\n7. Training (20 episodes)...")
    trainer = RLTrainer(config)
    trained_policy = trainer.train_on_window(env, n_episodes=20, verbose=False)
    print(f"   ✓ Complete")
    
    print("\n8. Evaluation...")
    results = trainer.evaluate_policy(env, trained_policy, deterministic=True)
    print(f"   ✓ Return: {results['total_return']:.6f}, Steps: {results['n_steps']}")
    
    print("\n9. Backtest...")
    rl_position, rl_strategy_selected = trainer.backtest_rl_policy(
        price_df, features_df, regime_df, strategy_positions, trained_policy
    )
    
    backtester = Backtester(config, initial_capital=10000.0)
    aligned_indices = regime_df.index.intersection(price_df.index)
    price_aligned = price_df.loc[aligned_indices]
    
    strategy_positions_aligned = {
        name: pos.loc[aligned_indices]
        for name, pos in strategy_positions.items()
    }
    
    all_results = backtester.backtest_all_models(
        price_aligned, regime_df, strategy_positions_aligned,
        rl_position=rl_position, rl_strategy_selected=rl_strategy_selected
    )
    
    print(f"   ✓ {len(all_results)} models")
    
    print("\n10. Metrics...")
    all_metrics = {}
    for name, result in all_results.items():
        metrics = calculate_all_metrics(
            result.equity_curve, result.returns,
            regime_labels=regime_df['regime_label'] if 'regime' in name.lower() or 'rl' in name.lower() else None,
            strategy_selected=result.strategy_selected
        )
        all_metrics[result.name] = metrics
    
    metrics_df = create_metrics_dataframe(all_metrics)
    
    print("\n" + "="*70)
    print("PERFORMANCE")
    print("="*70)
    print(metrics_df[['total_return', 'sharpe_ratio', 'max_drawdown']].to_string())
    
    print("\n" + "="*70)
    print("✅ PHASE 3 COMPLETE")
    print("="*70)
    print("\nComponents:")
    print("  ✓ RL environment")
    print("  ✓ Simple policy")  
    print("  ✓ Training loop")
    print("  ✓ Backtest integration")
    print("  ✓ Performance comparison")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)