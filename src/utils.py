import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


def set_random_seeds(seed: int = 42) -> None:

    np.random.seed(seed)
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def plot_regime_colored_price(
    price_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    price_column: str = 'close',
    figsize: tuple = (15, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:

    # Merge price and regime data
    plot_data = price_df[[price_column]].join(regime_df['regime_label'], how='inner')
    
    # Define colors for regimes
    n_regimes = plot_data['regime_label'].nunique()
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each regime with different color
    for regime in sorted(plot_data['regime_label'].unique()):
        mask = plot_data['regime_label'] == regime
        regime_data = plot_data[mask]
        
        ax.plot(
            regime_data.index,
            regime_data[price_column],
            color=colors[regime],
            label=f'Regime {regime}',
            linewidth=1.5
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title(title or 'Price Colored by Market Regime', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_regime_probabilities(
    regime_df: pd.DataFrame,
    figsize: tuple = (15, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:

    # Extract probability columns
    prob_cols = [col for col in regime_df.columns if col.startswith('regime_prob_')]
    prob_data = regime_df[prob_cols]
    
    # Sort columns by regime number
    prob_cols_sorted = sorted(prob_cols, key=lambda x: int(x.split('_')[-1]))
    prob_data = prob_data[prob_cols_sorted]
    
    # Create labels
    labels = [f"Regime {col.split('_')[-1]}" for col in prob_cols_sorted]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.stackplot(
        prob_data.index,
        prob_data.T.values,
        labels=labels,
        alpha=0.7
    )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title or 'Regime Probabilities Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_distributions_by_regime(
    features: pd.DataFrame,
    regime_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    figsize: tuple = (15, 10)
) -> None:

    if feature_cols is None:
        feature_cols = ['rolling_std', 'rolling_mean_return', 'rsi', 'ma_slope']
    
    # Merge features with regime labels
    merged = features.join(regime_df['regime_label'], how='inner')
    
    n_features = len(feature_cols)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        
        for regime in sorted(merged['regime_label'].unique()):
            regime_data = merged[merged['regime_label'] == regime][feature]
            ax.hist(regime_data, alpha=0.5, label=f'Regime {regime}', bins=50)
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions by Regime', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_regime_summary(regime_df: pd.DataFrame, window_info: list) -> None:

    print("\nREGIME DETECTION SUMMARY")
    print("*"*50)
    
    print(f"\nTotal bars with regime labels: {len(regime_df)}")
    print(f"Number of rolling windows: {len(window_info)}")
    
    print("\nRegime distribution:")
    regime_counts = regime_df['regime_label'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        print(f"  Regime {regime}: {count:6d} bars ({pct:5.2f}%)")
    
    print("\nWindow information:")
    print(f"  First window train start: {window_info[0]['train_start']}")
    print(f"  Last window test end: {window_info[-1]['test_end']}")
    print(f"  Average test window size: {np.mean([w['test_size'] for w in window_info]):.1f} bars")
    