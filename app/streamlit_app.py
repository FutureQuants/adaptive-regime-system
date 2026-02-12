import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src to path so all imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from config import Config
from data_loader import download_data_if_missing
from features import engineer_features
from regime import RegimeDetector, summarize_regimes
from strategies import generate_all_strategies
from backtester import Backtester
from rl_env import StrategySelectionEnv
from rl_trainer import RLTrainer
from metrics import calculate_all_metrics, create_metrics_dataframe
from utils import set_random_seeds


# Page configuration
st.set_page_config(
    page_title="Regime-Aware Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Fix for metric visibility */
    [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #31333F !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #09ab3b !important;
        font-size: 1rem !important;
    }
    
    /* Improve metric container background */
    [data-testid="metric-container"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(symbol, interval, start_date, end_date, n_regimes, 
                          train_days, test_days, random_seed):
    """Load and process all data (cached)."""
    
    set_random_seeds(random_seed)
    
    # Configuration
    config = Config(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        n_regimes=n_regimes,
        train_window_days=train_days,
        test_window_days=test_days,
        random_seed=random_seed
    )
    
    # Load data
    with st.spinner("📥 Loading data..."):
        df = download_data_if_missing(
            symbol=config.symbol,
            interval=config.interval,
            start_date=config.start_date,
            end_date=config.end_date,
            data_dir=config.data_dir
        )
    
    # Features
    with st.spinner("🔧 Engineering features..."):
        price_df, features_df = engineer_features(df, config)
    
    # Regimes
    with st.spinner("🎯 Detecting regimes..."):
        detector = RegimeDetector(config)
        regime_df, window_info = detector.detect_regimes(features_df, verbose=False)
    
    # Strategies
    with st.spinner("📊 Generating strategies..."):
        strategy_positions = generate_all_strategies(price_df, features_df, config)
    
    return config, price_df, features_df, regime_df, window_info, strategy_positions


@st.cache_data
def run_backtests(_config, _price_df, _regime_df, _strategy_positions):
    """Run all backtests (cached)."""
    
    backtester = Backtester(_config, initial_capital=10000.0)
    
    # Align data
    aligned_indices = _regime_df.index.intersection(_price_df.index)
    price_aligned = _price_df.loc[aligned_indices]
    
    strategy_positions_aligned = {
        name: pos.loc[aligned_indices]
        for name, pos in _strategy_positions.items()
    }
    
    # Run backtests
    results = backtester.backtest_all_models(
        price_aligned,
        _regime_df,
        strategy_positions_aligned
    )
    
    return results


def train_rl_policy(config, price_df, features_df, regime_df, strategy_positions, n_episodes):
    """Train RL policy."""
    
    trainer = RLTrainer(config)
    
    # Use subset for training
    test_regime_df = regime_df.iloc[:min(2000, len(regime_df))]
    
    env = StrategySelectionEnv(
        price_df, features_df, test_regime_df,
        strategy_positions, config, is_training=True
    )
    
    with st.spinner(f"🤖 Training RL policy ({n_episodes} episodes)..."):
        policy = trainer.train_on_window(env, n_episodes=n_episodes, verbose=False)
    
    # Backtest RL
    rl_position, rl_strategy_selected = trainer.backtest_rl_policy(
        price_df, features_df, regime_df, strategy_positions, policy
    )
    
    return policy, rl_position, rl_strategy_selected


def plot_price_with_regimes(price_df, regime_df):
    """Plot price colored by regime."""
    
    # Merge data
    plot_data = price_df[['close']].join(regime_df['regime_label'], how='inner')
    
    fig = go.Figure()
    
    # Plot each regime separately for coloring
    regime_colors = {0: '#2ecc71', 1: '#3498db', 2: '#e74c3c'}
    regime_names = {0: 'Low Volatility', 1: 'Medium Volatility', 2: 'High Volatility'}
    
    for regime in sorted(plot_data['regime_label'].unique()):
        mask = plot_data['regime_label'] == regime
        regime_data = plot_data[mask]
        
        fig.add_trace(go.Scatter(
            x=regime_data.index,
            y=regime_data['close'],
            mode='lines',
            name=regime_names.get(regime, f'Regime {regime}'),
            line=dict(color=regime_colors.get(regime, '#95a5a6'), width=2),
            hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Price History Colored by Market Regime',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_regime_probabilities(regime_df):
    """Plot regime probabilities as stacked area."""
    
    prob_cols = [col for col in regime_df.columns if col.startswith('regime_prob_')]
    prob_data = regime_df[sorted(prob_cols)]
    
    fig = go.Figure()
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, col in enumerate(sorted(prob_cols)):
        regime_num = col.split('_')[-1]
        fig.add_trace(go.Scatter(
            x=prob_data.index,
            y=prob_data[col],
            mode='lines',
            name=f'Regime {regime_num}',
            stackgroup='one',
            fillcolor=colors[i] if i < len(colors) else None,
            line=dict(width=0.5),
            hovertemplate='%{x}<br>Probability: %{y:.2%}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Regime Probabilities Over Time',
        xaxis_title='Date',
        yaxis_title='Probability',
        hovermode='x unified',
        height=400,
        showlegend=True,
        template='plotly_white',
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def plot_equity_curves(results):
    """Plot equity curves comparison."""
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, (name, result) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve,
            mode='lines',
            name=result.name,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='%{x}<br>Equity: $%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Equity Curves Comparison',
        xaxis_title='Date',
        yaxis_title='Equity (USD)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_drawdowns(results):
    """Plot drawdown comparison."""
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, (name, result) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=result.drawdown.index,
            y=result.drawdown * 100,
            mode='lines',
            name=result.name,
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tozeroy',
            hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Drawdown Comparison',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_strategy_selection(result):
    """Plot strategy selection over time."""
    
    if result.strategy_selected is None:
        return None
    
    strategy_names = ['Trend', 'Mean Rev', 'Breakout', 'Flat']
    strategy_colors = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
    
    fig = go.Figure()
    
    # Create segments for each strategy
    for strategy_idx in range(4):
        mask = result.strategy_selected == strategy_idx
        strategy_data = result.strategy_selected[mask]
        
        if len(strategy_data) > 0:
            fig.add_trace(go.Scatter(
                x=strategy_data.index,
                y=[strategy_idx] * len(strategy_data),
                mode='markers',
                name=strategy_names[strategy_idx],
                marker=dict(
                    color=strategy_colors[strategy_idx],
                    size=8,
                    symbol='square'
                ),
                hovertemplate='%{x}<br>Strategy: ' + strategy_names[strategy_idx] + '<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'Strategy Selection Over Time - {result.name}',
        xaxis_title='Date',
        yaxis_title='Strategy',
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=strategy_names
        ),
        hovermode='closest',
        height=300,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_regime_transition_matrix(regime_df):
    """Plot regime transition matrix."""
    
    labels = regime_df['regime_label'].values
    
    # Calculate transition matrix
    n_regimes = regime_df['regime_label'].nunique()
    transition_matrix = np.zeros((n_regimes, n_regimes))
    
    for i in range(len(labels) - 1):
        from_regime = int(labels[i])
        to_regime = int(labels[i + 1])
        transition_matrix[from_regime, to_regime] += 1
    
    # Normalize
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, 
                                  where=row_sums!=0, 
                                  out=np.zeros_like(transition_matrix))
    
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=[f'To Regime {i}' for i in range(n_regimes)],
        y=[f'From Regime {i}' for i in range(n_regimes)],
        colorscale='Blues',
        text=np.round(transition_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='From %{y}<br>To %{x}<br>Probability: %{z:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Regime Transition Matrix',
        height=400,
        template='plotly_white'
    )
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">📈 Regime-Aware Adaptive Trading System</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data parameters
        st.subheader("Data Parameters")
        symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h"], index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-31"))
        
        # Regime parameters
        st.subheader("Regime Detection")
        n_regimes = st.slider("Number of Regimes", 2, 5, 3)
        train_days = st.slider("Train Window (days)", 5, 60, 10)
        test_days = st.slider("Test Window (days)", 1, 14, 3)
        
        # RL parameters
        st.subheader("RL Training")
        enable_rl = st.checkbox("Enable RL Meta-Controller", value=False)
        if enable_rl:
            n_episodes = st.slider("Training Episodes", 10, 200, 50)
        
        random_seed = st.number_input("Random Seed", value=42, step=1)
        
        st.markdown("---")
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        try:
            # Load and process data
            config, price_df, features_df, regime_df, window_info, strategy_positions = \
                load_and_process_data(
                    symbol, interval, 
                    start_date.strftime("%Y-%m-%d"), 
                    end_date.strftime("%Y-%m-%d"),
                    n_regimes, train_days, test_days, random_seed
                )
            
            st.success(f"✅ Loaded {len(price_df):,} bars | "
                      f"Detected regimes for {len(regime_df):,} bars | "
                      f"{len(window_info)} rolling windows")
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", 
                "🎯 Regime Analysis", 
                "💹 Strategy Performance",
                "🤖 RL Meta-Controller",
                "📈 Detailed Metrics"
            ])
            
            # Tab 1: Overview
            with tab1:
                st.header("Market Overview")
                
                # Price chart
                fig_price = plot_price_with_regimes(price_df, regime_df)
                st.plotly_chart(fig_price, use_container_width=True)
                
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Total Bars</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{len(price_df):,}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Date Range</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{(price_df.index[-1] - price_df.index[0]).days} days</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Regime Bars</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{len(regime_df):,}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Rolling Windows</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{len(window_info)}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                price_change = (price_df['close'].iloc[-1] / price_df['close'].iloc[0] - 1) * 100
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Price Change</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{price_change:+.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Strategies</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{len(strategy_positions)}</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                volatility = features_df['rolling_std'].mean() * 100
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Avg Volatility</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{volatility:.3f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 10px; color: white;'>
                    <h3 style='margin: 0; font-size: 1.2rem;'>Regimes</h3>
                    <p style='margin: 5px 0 0 0; font-size: 2rem; font-weight: bold;'>{n_regimes}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tab 2: Regime Analysis
            with tab2:
                st.header("Regime Analysis")
                
                # Regime probabilities
                fig_probs = plot_regime_probabilities(regime_df)
                st.plotly_chart(fig_probs, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Regime distribution
                    st.subheader("Regime Distribution")
                    regime_counts = regime_df['regime_label'].value_counts().sort_index()
                    regime_pct = (regime_counts / len(regime_df) * 100).round(2)
                    
                    regime_dist_df = pd.DataFrame({
                        'Regime': [f'Regime {i}' for i in regime_counts.index],
                        'Bars': regime_counts.values,
                        'Percentage': [f"{p:.2f}%" for p in regime_pct.values]
                    })
                    st.dataframe(regime_dist_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Regime transition matrix
                    fig_transitions = plot_regime_transition_matrix(regime_df)
                    st.plotly_chart(fig_transitions, use_container_width=True)
                
                # Regime characteristics
                st.subheader("Regime Characteristics")
                regime_summary = summarize_regimes(regime_df, features_df)
                st.dataframe(regime_summary.round(4), use_container_width=True)
            
            # Tab 3: Strategy Performance
            with tab3:
                st.header("Strategy Performance Comparison")
                
                # Run backtests
                results = run_backtests(config, price_df, regime_df, strategy_positions)
                
                # Equity curves
                fig_equity = plot_equity_curves(results)
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Drawdowns
                fig_dd = plot_drawdowns(results)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Performance metrics
                st.subheader("Performance Metrics")
                all_metrics = {}
                for name, result in results.items():
                    metrics = calculate_all_metrics(
                        result.equity_curve,
                        result.returns,
                        regime_labels=regime_df['regime_label'] if 'regime' in name.lower() else None,
                        strategy_selected=result.strategy_selected
                    )
                    all_metrics[result.name] = metrics
                
                metrics_df = create_metrics_dataframe(all_metrics)
                
                # Display key metrics
                display_cols = ['total_return', 'annualized_return', 'sharpe_ratio', 
                               'max_drawdown', 'calmar_ratio']
                metrics_display = metrics_df[display_cols].copy()
                metrics_display.columns = ['Total Return', 'Annual Return', 'Sharpe Ratio',
                                          'Max Drawdown', 'Calmar Ratio']
                
                # Format percentages
                for col in ['Total Return', 'Annual Return', 'Max Drawdown']:
                    metrics_display[col] = metrics_display[col].apply(lambda x: f"{x*100:.2f}%")
                
                # Format ratios
                for col in ['Sharpe Ratio', 'Calmar Ratio']:
                    metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(metrics_display, use_container_width=True)
                
                # Strategy selection for regime mapping
                if 'regime_mapping' in results:
                    st.subheader("Regime Mapping Strategy Selection")
                    fig_strat = plot_strategy_selection(results['regime_mapping'])
                    if fig_strat:
                        st.plotly_chart(fig_strat, use_container_width=True)
            
            # Tab 4: RL Meta-Controller
            with tab4:
                st.header("RL Meta-Controller")
                
                if enable_rl:
                    # Train RL
                    policy, rl_position, rl_strategy_selected = train_rl_policy(
                        config, price_df, features_df, regime_df, 
                        strategy_positions, n_episodes
                    )
                    
                    st.success(f"✅ RL training complete ({n_episodes} episodes)")
                    
                    # Add RL to backtests
                    backtester = Backtester(config, initial_capital=10000.0)
                    aligned_indices = regime_df.index.intersection(price_df.index)
                    price_aligned = price_df.loc[aligned_indices]
                    
                    rl_result = backtester.backtest_rl_policy(
                        price_aligned, regime_df, strategy_positions,
                        rl_position, rl_strategy_selected
                    )
                    
                    # Performance comparison
                    st.subheader("RL vs Baselines")
                    
                    results_with_rl = results.copy()
                    results_with_rl['rl_meta'] = rl_result
                    
                    # Equity curve
                    fig_rl_equity = plot_equity_curves(results_with_rl)
                    st.plotly_chart(fig_rl_equity, use_container_width=True)
                    
                    # Strategy selection
                    st.subheader("RL Strategy Selection")
                    fig_rl_strat = plot_strategy_selection(rl_result)
                    if fig_rl_strat:
                        st.plotly_chart(fig_rl_strat, use_container_width=True)
                    
                    # RL metrics
                    rl_metrics = calculate_all_metrics(
                        rl_result.equity_curve,
                        rl_result.returns,
                        regime_labels=regime_df['regime_label'],
                        strategy_selected=rl_result.strategy_selected
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                            <h3 style='margin: 0; font-size: 1rem;'>RL Total Return</h3>
                            <p style='margin: 5px 0 0 0; font-size: 2.5rem; font-weight: bold;'>{rl_metrics['total_return']*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                            <h3 style='margin: 0; font-size: 1rem;'>RL Sharpe Ratio</h3>
                            <p style='margin: 5px 0 0 0; font-size: 2.5rem; font-weight: bold;'>{rl_metrics['sharpe_ratio']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                            <h3 style='margin: 0; font-size: 1rem;'>RL Max Drawdown</h3>
                            <p style='margin: 5px 0 0 0; font-size: 2.5rem; font-weight: bold;'>{rl_metrics['max_drawdown']*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        best_baseline = max(all_metrics.items(), key=lambda x: x[1]['total_return'])
                        delta = (rl_metrics['total_return'] - best_baseline[1]['total_return'])*100
                        delta_color = '#00ff00' if delta >= 0 else '#ff0000'
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                            <h3 style='margin: 0; font-size: 1rem;'>Best Baseline</h3>
                            <p style='margin: 5px 0 0 0; font-size: 2.5rem; font-weight: bold;'>{best_baseline[1]['total_return']*100:.2f}%</p>
                            <p style='margin: 5px 0 0 0; font-size: 1rem; color: {delta_color};'>Δ {delta:+.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Strategy usage
                    st.subheader("Strategy Usage Frequency")
                    usage_data = []
                    strategy_names = ['Trend Following', 'Mean Reversion', 'Volatility Breakout', 'Flat']
                    for i in range(4):
                        key = f'strategy_{i}_usage'
                        if key in rl_metrics:
                            usage_data.append({
                                'Strategy': strategy_names[i],
                                'Usage': f"{rl_metrics[key]:.2f}%"
                            })
                    
                    if usage_data:
                        st.dataframe(pd.DataFrame(usage_data), use_container_width=True, hide_index=True)
                
                else:
                    st.info("👆 Enable RL Meta-Controller in the sidebar to train and evaluate the RL agent")
                    
                    st.markdown("""
                    ### About RL Meta-Controller
                    
                    The RL agent learns to select which strategy to use based on:
                    - **Current regime** (detected by GMM)
                    - **Regime probabilities** (uncertainty estimates)
                    - **Market volatility** (rolling standard deviation)
                    - **Strategy persistence** (current active strategy)
                    
                    The agent is trained using REINFORCE (policy gradient) to maximize:
                    - Strategy returns
                    - Minimize transaction costs
                    - Minimize strategy switching
                    """)
            
            # Tab 5: Detailed Metrics
            with tab5:
                st.header("Detailed Metrics & Analysis")
                
                # Full metrics table
                st.subheader("Complete Metrics Comparison")
                st.dataframe(metrics_df.round(4), use_container_width=True)
                
                # Trade analysis
                st.subheader("Trade Analysis")
                
                for name, result in results.items():
                    if len(result.trades) > 0:
                        with st.expander(f"📋 {result.name} Trades ({len(result.trades)})"):
                            trades_display = result.trades.copy()
                            if len(trades_display) > 0:
                                trades_display['timestamp'] = pd.to_datetime(trades_display['timestamp'])
                                st.dataframe(trades_display, use_container_width=True, hide_index=True)
                
                # Window information
                st.subheader("Rolling Window Details")
                window_df = pd.DataFrame(window_info)
                st.dataframe(window_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Welcome message
        st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to begin")
        
        st.markdown("""
        ## Welcome to the Regime-Aware Adaptive Trading System
        
        This system combines:
        - **Regime Detection** using Gaussian Mixture Models
        - **Multiple Trading Strategies** (Trend, Mean Reversion, Breakout)
        - **RL Meta-Controller** for dynamic strategy selection
        
        ### Quick Start
        1. Select your symbol and date range in the sidebar
        2. Configure regime detection parameters
        3. Optionally enable RL training
        4. Click "Run Analysis"
        
        ### Features
        - 📊 Interactive price and regime visualization
        - 🎯 Regime analysis and transition matrices
        - 💹 Strategy performance comparison
        - 🤖 RL meta-controller training and evaluation
        - 📈 Comprehensive performance metrics
        """)


if __name__ == "__main__":
    main()