import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle


class SimplePolicy:
    
    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        np.random.seed(seed)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 2-layer network
        hidden_dim = 64
        
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        
        self.lr = 0.001
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, state @ self.W1 + self.b1)
        logits = h1 @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        probs = self.forward(state)
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.action_dim, p=probs)
        return int(action)
    
    def update(self, states: List[np.ndarray], actions: List[int], rewards: List[float]) -> float:
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # Compute returns
        gamma = 0.99
        returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Normalize
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        total_loss = 0
        
        for state, action, ret in zip(states, actions, returns):
            probs = self.forward(state)
            loss = -np.log(probs[action] + 1e-8) * ret
            total_loss += loss
            
            # Gradients
            grad_logits = probs.copy()
            grad_logits[action] -= 1
            grad_logits *= -ret
            
            h1 = np.maximum(0, state @ self.W1 + self.b1)
            grad_W2 = np.outer(h1, grad_logits)
            grad_b2 = grad_logits
            
            grad_h1 = grad_logits @ self.W2.T
            grad_h1[h1 <= 0] = 0
            grad_W1 = np.outer(state, grad_h1)
            grad_b1 = grad_h1
            
            # Update
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
        
        return total_loss / len(states)


class RLTrainer:
    # RL Trainer with walk-forward protocol
    
    def __init__(self, config):
        self.config = config
        self.policies = {}
    
    def train_on_window(self, env, n_episodes: int = 50, verbose: bool = True) -> SimplePolicy:
        state_dim = env.get_state_dim()
        action_dim = env.get_action_dim()
        
        policy = SimplePolicy(state_dim, action_dim, seed=self.config.random_seed)
        
        episode_returns = []
        
        for episode in range(n_episodes):
            states, actions, rewards = [], [], []
            
            state = env.reset()
            done = False
            
            while not done:
                action = policy.select_action(state, deterministic=False)
                next_state, reward, done, info = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
            
            loss = policy.update(states, actions, rewards)
            episode_returns.append(sum(rewards))
            
            if verbose and (episode + 1) % 10 == 0:
                recent = np.mean(episode_returns[-10:])
                print(f"    Ep {episode+1}/{n_episodes}: Avg={recent:.6f}")
        
        return policy
    
    def evaluate_policy(self, env, policy: SimplePolicy, deterministic: bool = True) -> Dict:
        state = env.reset()
        done = False
        
        actions_taken, rewards_received = [], []
        
        while not done:
            action = policy.select_action(state, deterministic=deterministic)
            next_state, reward, done, info = env.step(action)
            
            actions_taken.append(action)
            rewards_received.append(reward)
            state = next_state
        
        summary = env.get_episode_summary()
        summary['actions'] = actions_taken
        summary['rewards'] = rewards_received
        
        return summary
    
    def backtest_rl_policy(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        strategy_positions: Dict[str, pd.Series],
        policy: SimplePolicy
    ) -> Tuple[pd.Series, pd.Series]:
        
        # Run RL policy and return positions and strategy selections.
        from rl_env import StrategySelectionEnv
        
        # Create evaluation environment
        env = StrategySelectionEnv(
            price_df,
            features_df,
            regime_df,
            strategy_positions,
            self.config,
            is_training=False
        )
        
        # Evaluate
        results = self.evaluate_policy(env, policy, deterministic=True)
        actions = results['actions']
        
        # Convert to positions
        indices = regime_df.index
        combined_position = pd.Series(0, index=indices, dtype=int)
        strategy_selected = pd.Series(-1, index=indices, dtype=int)
        
        strategy_names = ['trend', 'mean_reversion', 'breakout', 'flat']
        
        for i, action in enumerate(actions):
            if i >= len(indices):
                break
            
            idx = indices[i]
            strategy_name = strategy_names[action]
            
            if strategy_name in strategy_positions and idx in strategy_positions[strategy_name].index:
                combined_position.loc[idx] = strategy_positions[strategy_name].loc[idx]
            
            strategy_selected.loc[idx] = action
        
        return combined_position, strategy_selected