import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class RegimeDetector:
    
    def __init__(self, config):

        self.config = config
        self.n_regimes = config.n_regimes
        self.train_window_bars = config.get_train_window_bars()
        self.test_window_bars = config.get_test_window_bars()
        
        # GMM parameters
        self.covariance_type = config.gmm_covariance_type
        self.n_init = config.gmm_n_init
        self.random_state = config.gmm_random_state
        
    def _sort_regimes_by_volatility(
        self,
        gmm: GaussianMixture,
        scaler: StandardScaler,
        features: pd.DataFrame
    ) -> np.ndarray:
        
        # Get cluster assignments for training data
        features_scaled = scaler.transform(features)
        labels = gmm.predict(features_scaled)
        
        # Compute mean volatility (rolling_std) per cluster
        features_with_labels = features.copy()
        features_with_labels['cluster'] = labels
        
        mean_volatility_per_cluster = features_with_labels.groupby('cluster')['rolling_std'].mean()
        
        # Sort clusters by volatility
        sorted_clusters = mean_volatility_per_cluster.sort_values().index.values
        
        # Create mapping: old_label -> new_label
        mapping = np.zeros(self.n_regimes, dtype=int)
        for new_label, old_label in enumerate(sorted_clusters):
            mapping[old_label] = new_label
            
        return mapping
    
    def detect_regimes(
        self,
        features: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, List[Dict]]:
   
        print("Starting rolling regime detection...")
        print(f"Training window: {self.train_window_bars} bars ({self.config.train_window_days} days)")
        print(f"Test window: {self.test_window_bars} bars ({self.config.test_window_days} days)")
        print(f"Number of regimes: {self.n_regimes}")
        
        n_bars = len(features)
        window_size = self.train_window_bars + self.test_window_bars
        
        # Initialize output arrays
        regime_labels = np.full(n_bars, -1, dtype=int)
        regime_probs = np.full((n_bars, self.n_regimes), np.nan)
        
        window_info = []
        window_count = 0
        
        # Rolling walk-forward
        start_idx = 0
        
        while start_idx + self.train_window_bars <= n_bars:
            # Define train and test indices
            train_end_idx = start_idx + self.train_window_bars
            test_end_idx = min(train_end_idx + self.test_window_bars, n_bars)
            
            train_indices = range(start_idx, train_end_idx)
            test_indices = range(train_end_idx, test_end_idx)
            
            if len(test_indices) == 0:
                break
            
            # Extract train and test features
            train_features = features.iloc[train_indices].copy()
            test_features = features.iloc[test_indices].copy()
            
            # Fit StandardScaler on training data only
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_features)
            test_scaled = scaler.transform(test_features)
            
            # Fit GMM on training data only
            gmm = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                random_state=self.random_state
            )
            gmm.fit(train_scaled)
            
            # Get regime mapping sorted by volatility
            label_mapping = self._sort_regimes_by_volatility(gmm, scaler, train_features)
            
            # Predict on test data
            test_labels_raw = gmm.predict(test_scaled)
            test_probs_raw = gmm.predict_proba(test_scaled)
            
            # Apply label mapping to ensure consistent ordering
            test_labels = label_mapping[test_labels_raw]
            
            # Reorder probabilities according to mapping
            test_probs = np.zeros_like(test_probs_raw)
            for old_label in range(self.n_regimes):
                new_label = label_mapping[old_label]
                test_probs[:, new_label] = test_probs_raw[:, old_label]
            
            # Store results
            regime_labels[test_indices] = test_labels
            regime_probs[test_indices] = test_probs
            
            # Store window information
            window_info.append({
                'window_num': window_count,
                'train_start': features.index[start_idx],
                'train_end': features.index[train_end_idx - 1],
                'test_start': features.index[train_end_idx],
                'test_end': features.index[test_end_idx - 1],
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'gmm_converged': gmm.converged_,
                'label_mapping': label_mapping.tolist()
            })
            
            window_count += 1
            
            if verbose:
                print(f"Window {window_count}: Train {features.index[start_idx]} to {features.index[train_end_idx-1]}, "
                      f"Test {features.index[train_end_idx]} to {features.index[test_end_idx-1]} "
                      f"({len(test_indices)} bars)")
            
            # Move to next window
            start_idx = train_end_idx
        
        print(f"\nCompleted {window_count} windows")
        
        # Create output DataFrame
        regime_df = pd.DataFrame(index=features.index)
        regime_df['regime_label'] = regime_labels
        
        for i in range(self.n_regimes):
            regime_df[f'regime_prob_{i}'] = regime_probs[:, i]
        
        # Remove bars without regime labels (initial training period)
        regime_df_clean = regime_df[regime_df['regime_label'] >= 0].copy()
        
        print(f"Regime labels assigned to {len(regime_df_clean)} bars")
        print(f"Regime distribution:")
        print(regime_df_clean['regime_label'].value_counts().sort_index())
        
        return regime_df_clean, window_info


def summarize_regimes(regime_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:

    # Merge regime labels with features
    merged = regime_df[['regime_label']].join(features, how='inner')
    
    # Compute statistics per regime
    summary = merged.groupby('regime_label').agg({
        'rolling_std': ['mean', 'std', 'min', 'max'],
        'rolling_mean_return': ['mean', 'std'],
        'rsi': ['mean', 'std'],
        'ma_slope': ['mean', 'std'],
        'volume_zscore': ['mean', 'std']
    })
    
    # Add count
    counts = merged.groupby('regime_label').size()
    summary['count'] = counts
    summary['percentage'] = (counts / len(merged) * 100).round(2)
    
    return summary