import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BacktestResult:
    
    name: str
    equity_curve: pd.Series
    returns: pd.Series
    drawdown: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    strategy_selected: Optional[pd.Series] = None
    
    def __repr__(self):
        return f"BacktestResult(name={self.name}, final_equity={self.equity_curve.iloc[-1]:.2f})"


# Import helper functions at module level
def get_strategy_by_index(index: int) -> str:
    # Map the strategy index to name
    from strategies import get_strategy_by_index as _get
    return _get(index)

def get_strategy_index(name: str) -> int:
    # Map strategy name to index
    from strategies import get_strategy_index as _get
    return _get(name)


class Backtester:
    
    def __init__(self, config, initial_capital: float = 10000.0):

        self.config = config
        self.initial_capital = initial_capital
        self.transaction_cost = config.transaction_cost
    
    def _calculate_equity_curve(
        self,
        price_df: pd.DataFrame,
        position: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:

        position = position.reindex(price_df.index, fill_value=0)
        
        # Initialize arrays
        equity = np.zeros(len(price_df))
        equity[0] = self.initial_capital
        
        returns = np.zeros(len(price_df))
        
        # Track trades
        trades = []
        
        # What we held during previous bar
        prev_position = 0
        
        for i in range(1, len(price_df)):
            # What position do we hold during THIS bar?
            # We execute the signal from previous close
            current_position = position.iloc[i-1]
            
            # Did position change?
            position_changed = (current_position != prev_position)
            
            # Calculate return for holding during this bar
            if prev_position == 1:
                # We were long during the previous->current transition
                price_return = (price_df['close'].iloc[i] / price_df['close'].iloc[i-1]) - 1
            else:
                price_return = 0.0
            
            # Apply transaction cost if position changed
            if position_changed:
                price_return -= self.transaction_cost
                
                # Record trade (happens at open of this bar)
                trades.append({
                    'timestamp': price_df.index[i],
                    'action': 'BUY' if current_position == 1 else 'SELL',
                    'position': current_position,
                    'price': price_df['open'].iloc[i],
                    'equity_before_trade': equity[i-1]
                })
            
            # Update equity
            returns[i] = price_return
            equity[i] = equity[i-1] * (1 + returns[i])
            
            # Remember what we're holding now
            prev_position = current_position
        
        # Create series with proper index
        equity_curve = pd.Series(equity, index=price_df.index)
        returns_series = pd.Series(returns, index=price_df.index)
        
        # Create trades DataFrame
        if trades:
            trades_df = pd.DataFrame(trades)
        else:
            trades_df = pd.DataFrame(columns=['timestamp', 'action', 'position', 'price', 'equity_before_trade'])
        
        return equity_curve, returns_series, trades_df
    
    def backtest_buy_and_hold(self, price_df: pd.DataFrame) -> BacktestResult:

        # Buy and hold: always position = 1
        position = pd.Series(1, index=price_df.index)
        
        equity_curve, returns, trades = self._calculate_equity_curve(price_df, position)
        
        from metrics import calculate_drawdown_series
        drawdown = calculate_drawdown_series(equity_curve)
        
        return BacktestResult(
            name="Buy & Hold",
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            positions=position,
            trades=trades
        )
    
    def backtest_fixed_strategy(
        self,
        price_df: pd.DataFrame,
        position: pd.Series,
        name: str
    ) -> BacktestResult:
  
        equity_curve, returns, trades = self._calculate_equity_curve(price_df, position)
        
        from metrics import calculate_drawdown_series
        drawdown = calculate_drawdown_series(equity_curve)
        
        return BacktestResult(
            name=name,
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            positions=position,
            trades=trades
        )
    
    def backtest_regime_mapping(
        self,
        price_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        strategy_positions: Dict[str, pd.Series],
        mapping: Optional[Dict[int, str]] = None
    ) -> BacktestResult:
    
        # Default mapping
        if mapping is None:
            mapping = {
                0: 'mean_reversion',  # Low volatility → mean reversion
                1: 'trend',           # Medium volatility → trend
                2: 'breakout'         # High volatility → breakout
            }
        
        # Align data
        aligned_indices = regime_df.index.intersection(price_df.index)
        regime_labels = regime_df.loc[aligned_indices, 'regime_label']
        
        # Initialize combined position and strategy tracking
        combined_position = pd.Series(0, index=price_df.index, dtype=int)
        strategy_selected = pd.Series(-1, index=price_df.index, dtype=int)
        
        # For each bar, select strategy based on regime
        for idx in aligned_indices:
            regime = regime_labels.loc[idx]
            strategy_name = mapping.get(regime, 'flat')
            
            # Get position from selected strategy
            if strategy_name in strategy_positions:
                strategy_pos = strategy_positions[strategy_name]
                if idx in strategy_pos.index:
                    combined_position.loc[idx] = strategy_pos.loc[idx]
            
            # Track which strategy was selected
            strategy_idx = get_strategy_index(strategy_name)
            strategy_selected.loc[idx] = strategy_idx
        
        # Forward fill positions for bars without regime labels
        combined_position = combined_position.replace(0, np.nan).ffill().fillna(0).astype(int)
        
        equity_curve, returns, trades = self._calculate_equity_curve(price_df, combined_position)
        
        from metrics import calculate_drawdown_series
        drawdown = calculate_drawdown_series(equity_curve)
        
        # Only include strategy_selected for bars with regime labels
        strategy_selected = strategy_selected[strategy_selected >= 0]
        
        return BacktestResult(
            name="Regime Mapping",
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            positions=combined_position,
            trades=trades,
            strategy_selected=strategy_selected
        )
    
    def backtest_rl_strategy_selection(
        self,
        price_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        strategy_positions: Dict[str, pd.Series],
        rl_selections: pd.Series
    ) -> BacktestResult:
  
        # Align data
        aligned_indices = rl_selections.index.intersection(price_df.index)
        
        # Initialize combined position and strategy tracking
        combined_position = pd.Series(0, index=price_df.index, dtype=int)
        strategy_selected = pd.Series(-1, index=price_df.index, dtype=int)
        
        # For each bar, use RL-selected strategy
        for idx in aligned_indices:
            strategy_idx = rl_selections.loc[idx]
            strategy_name = get_strategy_by_index(strategy_idx)
            
            # Get position from selected strategy
            if strategy_name in strategy_positions:
                strategy_pos = strategy_positions[strategy_name]
                if idx in strategy_pos.index:
                    combined_position.loc[idx] = strategy_pos.loc[idx]
            
            # Track which strategy was selected
            strategy_selected.loc[idx] = strategy_idx
        
        # Forward fill positions for bars without RL selections
        combined_position = combined_position.replace(0, np.nan).ffill().fillna(0).astype(int)
        
        equity_curve, returns, trades = self._calculate_equity_curve(price_df, combined_position)
        
        from metrics import calculate_drawdown_series
        drawdown = calculate_drawdown_series(equity_curve)
        
        # Only include strategy_selected for bars with RL selections
        strategy_selected = strategy_selected[strategy_selected >= 0]
        
        return BacktestResult(
            name="RL Meta-Controller",
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            positions=combined_position,
            trades=trades,
            strategy_selected=strategy_selected
        )
    
    def backtest_rl_policy(
        self,
        price_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        strategy_positions: Dict[str, pd.Series],
        rl_position: pd.Series,
        rl_strategy_selected: pd.Series
    ) -> BacktestResult:
        
        equity_curve, returns, trades = self._calculate_equity_curve(price_df, rl_position)
        
        from metrics import calculate_drawdown_series
        drawdown = calculate_drawdown_series(equity_curve)
        
        return BacktestResult(
            name="RL Meta-Controller",
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            positions=rl_position,
            trades=trades,
            strategy_selected=rl_strategy_selected
        )
    
    def backtest_all_models(
        self,
        price_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        strategy_positions: Dict[str, pd.Series],
        rl_position: Optional[pd.Series] = None,
        rl_strategy_selected: Optional[pd.Series] = None
    ) -> Dict[str, BacktestResult]:

        results = {}
        
        print("Running backtests...")
        
        # Buy and Hold
        print("  - Buy & Hold")
        results['buy_hold'] = self.backtest_buy_and_hold(price_df)
        
        # Individual strategies
        strategy_names = {
            'trend': 'Trend Following',
            'mean_reversion': 'Mean Reversion',
            'breakout': 'Volatility Breakout',
            'flat': 'Flat'
        }
        
        for key, name in strategy_names.items():
            print(f"  - {name}")
            results[key] = self.backtest_fixed_strategy(
                price_df,
                strategy_positions[key],
                name
            )
        
        # Regime mapping
        print("  - Regime Mapping")
        results['regime_mapping'] = self.backtest_regime_mapping(
            price_df,
            regime_df,
            strategy_positions
        )
        
        # RL policy (if provided)
        if rl_position is not None and rl_strategy_selected is not None:
            print("  - RL Meta-Controller")
            results['rl_meta'] = self.backtest_rl_policy(
                price_df,
                regime_df,
                strategy_positions,
                rl_position,
                rl_strategy_selected
            )
        
        print("Backtests complete!")
        
        return results


def run_walk_forward_backtest(
    price_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    strategy_positions: Dict[str, pd.Series],
    config,
    initial_capital: float = 10000.0
) -> Dict[str, BacktestResult]:

    backtester = Backtester(config, initial_capital)
    
    # Align all data to regime indices (out-of-sample only)
    aligned_indices = regime_df.index.intersection(price_df.index)
    price_aligned = price_df.loc[aligned_indices]
    
    # Align strategy positions to out-of-sample period
    strategy_positions_aligned = {}
    for name, pos in strategy_positions.items():
        strategy_positions_aligned[name] = pos.loc[aligned_indices]
    
    # Run backtests on aligned data
    results = backtester.backtest_all_models(
        price_aligned,
        regime_df,
        strategy_positions_aligned
    )
    
    return results