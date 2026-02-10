"""
Charts: Plotly visualizations for backtest results.
Provides reusable chart generators for the UI and tearsheet.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .suite import SuiteResult


def plot_lag_sensitivity(suite_result: SuiteResult, ic_by_lag: Optional[Dict[int, float]] = None, show_turnover: bool = False) -> go.Figure:
    """
    Plot how key metrics decay as lag increases.
    
    Shows Sharpe ratio and optionally turnover/IC across different lag values.
    Highlights if signal "dies" with lag (Sharpe drops >50%).
    
    Args:
        suite_result: Result from run_suite() containing multiple lag configs
        ic_by_lag: Optional dict of {lag: IC} values (computed separately)
        show_turnover: Whether to show turnover on secondary y-axis (default False)
        
    Returns:
        Plotly figure with Sharpe on left y-axis, optionally turnover on right
    """
    # Extract lag and metrics from results
    data = []
    for config_key, result in suite_result.results.items():
        # Parse lag from config key (e.g., "lag0_residoff" -> 0)
        if 'lag' in config_key:
            parts = config_key.split('_')
            lag_part = [p for p in parts if p.startswith('lag')]
            if lag_part:
                lag = int(lag_part[0].replace('lag', ''))
                resid = config_key.split('_resid')[-1] if '_resid' in config_key else 'off'
                data.append({
                    'lag': lag,
                    'resid': resid,
                    'sharpe': result.sharpe,
                    'turnover': result.turnover,
                    'config_key': config_key,
                })
    
    if not data:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text="No lag data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(data)
    
    # Get unique resid options for coloring
    resid_options = df['resid'].unique()
    
    # Create figure with secondary y-axis if showing turnover
    fig = make_subplots(specs=[[{"secondary_y": show_turnover}]])
    
    colors = px.colors.qualitative.Set1
    
    for i, resid in enumerate(resid_options):
        subset = df[df['resid'] == resid].sort_values('lag')
        color = colors[i % len(colors)]
        
        # Sharpe (left y-axis)
        fig.add_trace(
            go.Scatter(
                x=subset['lag'], 
                y=subset['sharpe'],
                name=f'Sharpe ({resid})',
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=10),
            ),
            secondary_y=False
        )
        
        # Turnover (right y-axis) - dashed line (optional)
        if show_turnover:
            fig.add_trace(
                go.Scatter(
                    x=subset['lag'], 
                    y=subset['turnover'] * 100,  # Convert to percentage
                    name=f'Turnover ({resid})',
                    mode='lines+markers',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond'),
                ),
                secondary_y=True
            )
    
    # Add IC if provided
    if ic_by_lag:
        ic_df = pd.DataFrame([{'lag': k, 'ic': v} for k, v in ic_by_lag.items()]).sort_values('lag')
        fig.add_trace(
            go.Scatter(
                x=ic_df['lag'],
                y=ic_df['ic'],
                name='IC',
                mode='lines+markers',
                line=dict(color='green', width=2, dash='dot'),
                marker=dict(size=8, symbol='square'),
            ),
            secondary_y=False
        )
    
    # Check for "signal death" (Sharpe drops >50% from lag 0)
    lag0_sharpe = df[df['lag'] == 0]['sharpe'].max() if 0 in df['lag'].values else None
    if lag0_sharpe:
        max_lag = df['lag'].max()
        max_lag_sharpe = df[df['lag'] == max_lag]['sharpe'].max()
        if max_lag_sharpe < lag0_sharpe * 0.5:
            fig.add_annotation(
                text="Signal decays >50% with lag",
                xref="paper", yref="paper",
                x=0.95, y=0.95,
                showarrow=False,
                font=dict(color="red", size=12),
                bgcolor="rgba(255,200,200,0.8)",
                bordercolor="red",
                borderwidth=1,
            )
    
    # Update layout
    title = 'Lag Sensitivity: Sharpe & Turnover vs Lag' if show_turnover else 'Lag Sensitivity: Sharpe vs Lag'
    fig.update_layout(
        title=title,
        xaxis_title='Lag (trading days)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified',
    )
    
    fig.update_yaxes(title_text="Sharpe Ratio / IC", secondary_y=False)
    if show_turnover:
        fig.update_yaxes(title_text="Turnover (%)", secondary_y=True)
    
    return fig


def plot_lag_sensitivity_from_summary(summary_df: pd.DataFrame, show_turnover: bool = False) -> go.Figure:
    """
    Plot lag sensitivity from a summary DataFrame (for History tab).
    
    Args:
        summary_df: Summary DataFrame with 'config', 'sharpe', 'turnover' columns
        show_turnover: Whether to show turnover on secondary y-axis
        
    Returns:
        Plotly figure
    """
    if summary_df is None or summary_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Filter to signal configs only
    if 'type' in summary_df.columns:
        df = summary_df[summary_df['type'] == 'signal'].copy()
    else:
        df = summary_df.copy()
    
    # Extract lag and resid from config column
    data = []
    for _, row in df.iterrows():
        config_key = row.get('config', '')
        if 'lag' in str(config_key):
            parts = str(config_key).split('_')
            lag_part = [p for p in parts if p.startswith('lag')]
            if lag_part:
                lag = int(lag_part[0].replace('lag', ''))
                resid = str(config_key).split('_resid')[-1] if '_resid' in str(config_key) else 'off'
                data.append({
                    'lag': lag,
                    'resid': resid,
                    'sharpe': row.get('sharpe', 0),
                    'turnover': row.get('turnover', 0),
                })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No lag data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    plot_df = pd.DataFrame(data)
    resid_options = plot_df['resid'].unique()
    
    fig = make_subplots(specs=[[{"secondary_y": show_turnover}]])
    colors = px.colors.qualitative.Set1
    
    for i, resid in enumerate(resid_options):
        subset = plot_df[plot_df['resid'] == resid].sort_values('lag')
        color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=subset['lag'], 
                y=subset['sharpe'],
                name=f'Sharpe ({resid})',
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=10),
            ),
            secondary_y=False
        )
        
        if show_turnover:
            fig.add_trace(
                go.Scatter(
                    x=subset['lag'], 
                    y=subset['turnover'] * 100,
                    name=f'Turnover ({resid})',
                    mode='lines+markers',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond'),
                ),
                secondary_y=True
            )
    
    title = 'Lag Sensitivity: Sharpe & Turnover vs Lag' if show_turnover else 'Lag Sensitivity: Sharpe vs Lag'
    fig.update_layout(
        title=title,
        xaxis_title='Lag (trading days)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified',
    )
    
    fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=False)
    if show_turnover:
        fig.update_yaxes(title_text="Turnover (%)", secondary_y=True)
    
    return fig


def get_lag_sensitivity_table(suite_result: SuiteResult, ic_by_lag: Optional[Dict[int, float]] = None) -> pd.DataFrame:
    """
    Create a summary table of metrics by lag for tearsheet.
    
    Returns DataFrame with columns: Lag, Sharpe, Turnover, IC (if provided)
    """
    data = []
    for config_key, result in suite_result.results.items():
        if 'lag' in config_key:
            parts = config_key.split('_')
            lag_part = [p for p in parts if p.startswith('lag')]
            if lag_part:
                lag = int(lag_part[0].replace('lag', ''))
                resid = config_key.split('_resid')[-1] if '_resid' in config_key else 'off'
                row = {
                    'Lag': lag,
                    'Resid': resid,
                    'Sharpe': result.sharpe,
                    'Turnover': result.turnover,
                }
                if ic_by_lag and lag in ic_by_lag:
                    row['IC'] = ic_by_lag[lag]
                data.append(row)
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data).sort_values(['Lag', 'Resid'])
    return df


def plot_cumulative_pnl(daily_df: pd.DataFrame, title: str = "Cumulative Return") -> go.Figure:
    """
    Plot cumulative PnL over time.
    
    Args:
        daily_df: DataFrame with 'date' and 'cumret' columns.
                  Optionally 'config' for multiple series.
    """
    if 'config' in daily_df.columns:
        fig = px.line(daily_df, x='date', y='cumret', color='config', title=title)
    else:
        fig = px.line(daily_df, x='date', y='cumret', title=title)
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
    )
    return fig


def plot_drawdown(daily_df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    """
    Plot drawdown over time.
    
    Args:
        daily_df: DataFrame with 'date' and 'drawdown' columns.
    """
    if 'drawdown' not in daily_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No drawdown data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df['date'],
        y=daily_df['drawdown'] * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
    )
    return fig


def plot_decile_returns(fractile_df: pd.DataFrame, title: str = "Returns by Decile") -> go.Figure:
    """
    Plot bar chart of returns by signal decile.
    Highlights monotonicity (ideal: increasing from decile 1 to 10).
    
    Args:
        fractile_df: DataFrame with 'fractile' and 'ret' (or 'resret') columns.
    """
    if fractile_df is None or fractile_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No fractile data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Use raw returns ('ret') by default for consistency with portfolio returns
    y_col = 'ret' if 'ret' in fractile_df.columns else 'resret'
    
    # Sort by fractile
    df = fractile_df.sort_values('fractile').copy()
    
    # Demean returns to show excess returns relative to universe average
    # This makes the chart more meaningful: positive = outperform, negative = underperform
    universe_mean = df[y_col].mean()
    df['excess_ret'] = df[y_col] - universe_mean
    
    # Color bars based on excess return (green for outperform, red for underperform)
    colors = ['green' if v > 0 else 'red' for v in df['excess_ret']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['fractile'],
        y=df['excess_ret'] * 100,  # Convert to percentage
        marker_color=colors,
        text=[f"{v*100:+.1f}%" for v in df['excess_ret']],
        textposition='outside',
    ))
    
    # Check monotonicity (using excess returns)
    returns = df['excess_ret'].values
    is_monotonic = all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
    
    # Calculate spread (D10 - D1)
    spread = returns[-1] - returns[0] if len(returns) >= 2 else 0
    
    if is_monotonic:
        fig.add_annotation(
            text=f"Monotonic | Spread: {spread*100:.1f}%",
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            font=dict(color="green", size=12),
            bgcolor="rgba(200,255,200,0.8)",
        )
    else:
        # Check if spread is meaningful (D10 - D1 > 0)
        if spread > 0:
            fig.add_annotation(
                text=f"Spread: {spread*100:.1f}%",
                xref="paper", yref="paper",
                x=0.05, y=0.95,
                showarrow=False,
                font=dict(color="blue", size=12),
            )
    
    fig.update_layout(
        title=title,
        xaxis_title='Signal Decile',
        yaxis_title='Excess Return vs Universe (%)',
        showlegend=False,
    )
    
    return fig


def plot_factor_exposure_bars(factor_exposures: pd.DataFrame, title: str = "Risk Factor Exposures") -> go.Figure:
    """
    Plot horizontal bar chart of signal correlation to risk factors.
    
    Args:
        factor_exposures: DataFrame with 'factor' and 'correlation' columns,
                         or Series with factor names as index.
    """
    if factor_exposures is None or (isinstance(factor_exposures, pd.DataFrame) and factor_exposures.empty):
        fig = go.Figure()
        fig.add_annotation(text="No factor exposure data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Convert to DataFrame if Series
    if isinstance(factor_exposures, pd.Series):
        df = factor_exposures.reset_index()
        df.columns = ['factor', 'correlation']
    elif 'factor' in factor_exposures.columns and 'correlation' in factor_exposures.columns:
        df = factor_exposures
    else:
        # Assume first column is factor, second is correlation
        df = factor_exposures.iloc[:, :2].copy()
        df.columns = ['factor', 'correlation']
    
    # Color based on correlation sign and magnitude
    def get_color(corr):
        if abs(corr) > 0.3:
            return 'red' if corr > 0 else 'darkred'  # High exposure warning
        elif corr > 0:
            return 'lightblue'
        else:
            return 'lightsalmon'
    
    colors = [get_color(c) for c in df['correlation']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['factor'],
        x=df['correlation'],
        orientation='h',
        marker_color=colors,
        text=[f"{c:.3f}" for c in df['correlation']],
        textposition='outside',
    ))
    
    # Add reference lines
    fig.add_vline(x=0, line_dash="solid", line_color="gray")
    fig.add_vline(x=0.3, line_dash="dash", line_color="red", annotation_text="High", annotation_position="top")
    fig.add_vline(x=-0.3, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title='Correlation',
        yaxis_title='',
        showlegend=False,
        xaxis=dict(range=[-0.5, 0.5]),
    )
    
    return fig


def plot_coverage_over_time(signal_df: pd.DataFrame, title: str = "Signal Coverage Over Time") -> go.Figure:
    """
    Plot number of securities with signal over time.
    
    Args:
        signal_df: Signal DataFrame with 'date_sig' and 'security_id' columns.
    """
    if signal_df is None or signal_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No signal data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Count securities per day
    coverage = signal_df.groupby('date_sig')['security_id'].nunique().reset_index()
    coverage.columns = ['date', 'count']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coverage['date'],
        y=coverage['count'],
        mode='lines',
        fill='tozeroy',
        name='Securities with Signal',
        line=dict(color='steelblue'),
    ))
    
    # Add average line
    avg_count = coverage['count'].mean()
    fig.add_hline(y=avg_count, line_dash="dash", line_color="orange",
                  annotation_text=f"Avg: {avg_count:.0f}", annotation_position="right")
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Securities',
        showlegend=False,
        hovermode='x unified',
    )
    
    return fig


def plot_ic_over_time(ic_series: pd.DataFrame, title: str = "Information Coefficient Over Time") -> go.Figure:
    """
    Plot IC time series with rolling average.
    
    Args:
        ic_series: DataFrame with 'date' and 'ic' columns.
    """
    if ic_series is None or ic_series.empty:
        fig = go.Figure()
        fig.add_annotation(text="No IC data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = ic_series.sort_values('date')
    
    # Add rolling average (21-day)
    df['ic_ma'] = df['ic'].rolling(21, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Daily IC (light)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ic'],
        mode='lines',
        name='Daily IC',
        line=dict(color='lightgray', width=1),
    ))
    
    # Rolling average (bold)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ic_ma'],
        mode='lines',
        name='21-day MA',
        line=dict(color='blue', width=2),
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray")
    
    # Add mean IC annotation
    mean_ic = df['ic'].mean()
    fig.add_annotation(
        text=f"Mean IC: {mean_ic:.4f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Information Coefficient',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
    )
    
    return fig
