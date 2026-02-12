import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().fillna(0)


def calculate_total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) == 0:
        return 0.0
    
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0


def calculate_annualized_return(
    equity_curve: pd.Series,
    periods_per_year: int = 525600  # 1-minute bars
) -> float:

    if len(equity_curve) < 2:
        return 0.0
    
    total_return = calculate_total_return(equity_curve)
    n_periods = len(equity_curve)
    
    if n_periods == 0:
        return 0.0
    
    years = n_periods / periods_per_year
    
    if years == 0:
        return 0.0
    
    annualized = (1 + total_return) ** (1 / years) - 1
    
    return annualized


def calculate_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 525600
) -> float:

    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 525600,
    risk_free_rate: float = 0.0
) -> float:

    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_max_drawdown(equity_curve: pd.Series) -> float:

    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Return maximum drawdown (as positive value)
    max_dd = abs(drawdown.min())
    
    return max_dd


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:

    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:

    if max_drawdown == 0:
        return 0.0
    
    return annualized_return / max_drawdown


def calculate_strategy_usage_frequency(
    strategy_selected: pd.Series
) -> Dict[int, float]:

    counts = strategy_selected.value_counts()
    total = len(strategy_selected)
    
    frequencies = {}
    for strategy_idx in range(4):
        frequencies[strategy_idx] = (counts.get(strategy_idx, 0) / total * 100) if total > 0 else 0.0
    
    return frequencies


def calculate_regime_conditional_sharpe(
    returns: pd.Series,
    regime_labels: pd.Series,
    periods_per_year: int = 525600
) -> Dict[int, float]:

    # Align returns and regimes
    aligned = pd.DataFrame({
        'returns': returns,
        'regime': regime_labels
    }).dropna()
    
    regime_sharpe = {}
    
    for regime in sorted(aligned['regime'].unique()):
        regime_returns = aligned[aligned['regime'] == regime]['returns']
        
        if len(regime_returns) > 1:
            sharpe = calculate_sharpe_ratio(regime_returns, periods_per_year)
        else:
            sharpe = 0.0
        
        regime_sharpe[int(regime)] = sharpe
    
    return regime_sharpe


def calculate_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    regime_labels: Optional[pd.Series] = None,
    strategy_selected: Optional[pd.Series] = None,
    periods_per_year: int = 525600
) -> Dict[str, float]:
  
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = calculate_total_return(equity_curve)
    metrics['annualized_return'] = calculate_annualized_return(equity_curve, periods_per_year)
    metrics['annualized_volatility'] = calculate_annualized_volatility(returns, periods_per_year)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, periods_per_year)
    metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)
    metrics['calmar_ratio'] = calculate_calmar_ratio(
        metrics['annualized_return'],
        metrics['max_drawdown']
    )
    
    # Strategy usage frequency (if applicable)
    if strategy_selected is not None:
        usage = calculate_strategy_usage_frequency(strategy_selected)
        for idx, freq in usage.items():
            metrics[f'strategy_{idx}_usage'] = freq
    
    # Regime-conditional Sharpe (if applicable)
    if regime_labels is not None:
        regime_sharpe = calculate_regime_conditional_sharpe(
            returns,
            regime_labels,
            periods_per_year
        )
        for regime, sharpe in regime_sharpe.items():
            metrics[f'regime_{regime}_sharpe'] = sharpe
    
    return metrics


def create_metrics_dataframe(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:

    df = pd.DataFrame(metrics_dict).T
    
    # Round for display
    for col in df.columns:
        if 'return' in col or 'ratio' in col or 'volatility' in col or 'drawdown' in col:
            df[col] = df[col].round(4)
        elif 'usage' in col:
            df[col] = df[col].round(2)
    
    return df