import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class StrategySelectionEnv:

    def __init__(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        strategy_positions: Dict[str, pd.Series],
        config,
        is_training: bool = True
    ):
  
        self.config = config
        self.is_training = is_training
        
        # Align all data to regime indices (out-of-sample only)
        self.indices = regime_df.index.intersection(price_df.index)
        self.price_df = price_df.loc[self.indices].copy()
        self.features_df = features_df.loc[self.indices].copy()
        self.regime_df = regime_df.loc[self.indices].copy()
        
        # Align strategy positions
        self.strategy_positions = {}
        for name, pos in strategy_positions.items():
            self.strategy_positions[name] = pos.loc[self.indices].values
        
        # Environment parameters
        self.n_regimes = config.n_regimes
        self.n_strategies = 4
        self.transaction_cost = config.transaction_cost
        self.switching_penalty = config.switching_penalty
        
        # State tracking
        self.current_step = 0
        self.current_strategy = 3  # Start with Flat
        self.current_position = 0
        self.total_steps = len(self.indices)
        
        # Episode tracking
        self.episode_returns = []
        self.episode_actions = []
        
    def _get_state(self) -> np.ndarray:

        # Regime probabilities
        regime_probs = np.zeros(self.n_regimes)
        for i in range(self.n_regimes):
            col_name = f'regime_prob_{i}'
            if col_name in self.regime_df.columns:
                regime_probs[i] = self.regime_df.iloc[self.current_step][col_name]
        
        # Current regime (one-hot)
        regime_label = int(self.regime_df.iloc[self.current_step]['regime_label'])
        regime_onehot = np.zeros(self.n_regimes)
        regime_onehot[regime_label] = 1.0
        
        # Rolling volatility (normalized)
        volatility = self.features_df.iloc[self.current_step]['rolling_std']
        # Normalize to [0, 1] range using reasonable bounds
        vol_normalized = np.clip(volatility / 0.01, 0, 1)  # Assuming 1% std as upper bound
        
        # Current strategy (one-hot)
        strategy_onehot = np.zeros(self.n_strategies)
        strategy_onehot[self.current_strategy] = 1.0
        
        # Concatenate all components
        state = np.concatenate([
            regime_probs,
            regime_onehot,
            [vol_normalized],
            strategy_onehot
        ])
        
        return state.astype(np.float32)
    
    def _get_strategy_name(self, action: int) -> str:
        # Map action index to strategy name
        mapping = {
            0: 'trend',
            1: 'mean_reversion',
            2: 'breakout',
            3: 'flat'
        }
        return mapping.get(action, 'flat')
    
    def _calculate_reward(
        self,
        action: int,
        previous_strategy: int,
        previous_position: int
    ) -> float:
   
        # Get current position from selected strategy
        strategy_name = self._get_strategy_name(action)
        current_position = self.strategy_positions[strategy_name][self.current_step]
        
        # Calculate return if we were long
        if previous_position == 1:
            # Return from previous close to current open
            price_return = (
                self.price_df.iloc[self.current_step]['open'] /
                self.price_df.iloc[self.current_step - 1]['close']
            ) - 1.0
        else:
            price_return = 0.0
        
        # Apply transaction cost if position changed
        position_changed = (current_position != previous_position)
        if position_changed:
            price_return -= self.transaction_cost
        
        # Apply switching penalty if strategy changed
        strategy_changed = (action != previous_strategy)
        if strategy_changed:
            price_return -= self.switching_penalty
        
        return price_return
    
    def reset(self) -> np.ndarray:
  
        self.current_step = 1  # Start at 1 to allow lookback
        self.current_strategy = 3  # Start with Flat
        self.current_position = 0
        self.episode_returns = []
        self.episode_actions = []
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
    
        # Validate action
        action = int(np.clip(action, 0, self.n_strategies - 1))
        
        # Calculate reward
        reward = self._calculate_reward(
            action,
            self.current_strategy,
            self.current_position
        )
        
        # Update state
        strategy_name = self._get_strategy_name(action)
        new_position = self.strategy_positions[strategy_name][self.current_step]
        
        # Track episode
        self.episode_returns.append(reward)
        self.episode_actions.append(action)
        
        # Update current state
        self.current_strategy = action
        self.current_position = new_position
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= self.total_steps - 1)
        
        # Get next state
        if not done:
            next_state = self._get_state()
        else:
            next_state = self._get_state()  # Return final state
        
        # Info dictionary
        info = {
            'strategy_selected': action,
            'position': new_position,
            'cumulative_return': sum(self.episode_returns)
        }
        
        return next_state, reward, done, info
    
    def get_state_dim(self) -> int:
        # Get dimension of state space
        # regime_probs + regime_onehot + volatility + strategy_onehot
        return self.n_regimes + self.n_regimes + 1 + self.n_strategies
    
    def get_action_dim(self) -> int:
        # Get dimension of action space
        return self.n_strategies
    
    def get_episode_summary(self) -> Dict:

        if len(self.episode_returns) == 0:
            return {
                'total_return': 0.0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'strategy_usage': {i: 0.0 for i in range(self.n_strategies)}
            }
        
        # Strategy usage frequency
        strategy_usage = {}
        total_actions = len(self.episode_actions)
        for i in range(self.n_strategies):
            count = sum(1 for a in self.episode_actions if a == i)
            strategy_usage[i] = (count / total_actions * 100) if total_actions > 0 else 0.0
        
        return {
            'total_return': sum(self.episode_returns),
            'mean_reward': np.mean(self.episode_returns),
            'std_reward': np.std(self.episode_returns),
            'n_steps': len(self.episode_returns),
            'strategy_usage': strategy_usage
        }