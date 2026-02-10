"""
Tear Sheet Generator: Create HTML reports from suite results.
~200 lines - includes verdict logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from jinja2 import Template

from .suite import SuiteResult, get_best_config, RISK_FACTORS
from .charts import get_lag_sensitivity_table


# Simple HTML template (inline to avoid file management)
TEARSHEET_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Tear Sheet: {{ signal_name }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: right; }
        th { background: #f8f9fa; font-weight: 600; }
        tr:nth-child(even) { background: #f8f9fa; }
        .verdict { padding: 20px; border-radius: 8px; margin: 20px 0; }
        .verdict.green { background: #d4edda; border: 1px solid #c3e6cb; }
        .verdict.yellow { background: #fff3cd; border: 1px solid #ffeeba; }
        .verdict.red { background: #f8d7da; border: 1px solid #f5c6cb; }
        .verdict h3 { margin-top: 0; }
        .metric-card { display: inline-block; padding: 15px 25px; margin: 10px; background: #f8f9fa; border-radius: 8px; text-align: center; }
        .metric-card .value { font-size: 28px; font-weight: bold; color: #007bff; }
        .metric-card .label { font-size: 12px; color: #666; text-transform: uppercase; }
        .section { margin: 30px 0; }
        .timestamp { color: #999; font-size: 12px; }
        .subsection { margin: 20px 0; }
        .subsection h3 { color: #666; font-size: 16px; margin-bottom: 10px; }
        .two-column { display: flex; gap: 30px; }
        .two-column > div { flex: 1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ signal_name }}</h1>
        <p class="timestamp">Generated: {{ timestamp }} | Snapshot: {{ snapshot_id }}</p>
        
        <!-- Verdict Panel -->
        <div class="verdict {{ verdict.color }}">
            <h3>Verdict: {{ verdict.color | upper }}</h3>
            {% if composite_score %}
            <div style="float: right; text-align: center; margin-left: 20px;">
                <div style="font-size: 48px; font-weight: bold; color: {% if composite_score.grade == 'A' %}#28a745{% elif composite_score.grade == 'B' %}#17a2b8{% elif composite_score.grade == 'C' %}#ffc107{% else %}#dc3545{% endif %};">
                    {{ composite_score.grade }}
                </div>
                <div style="font-size: 14px; color: #666;">{{ "%.0f"|format(composite_score.total_score) }}/100</div>
            </div>
            {% endif %}
            <ul>
            {% for reason in verdict.reasons %}
                <li>{{ reason }}</li>
            {% endfor %}
            </ul>
        </div>
        
        {% if composite_score and composite_score.breakdown %}
        <!-- Quality Score Breakdown -->
        <div class="section">
            <h2>Quality Score Breakdown</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Score</th>
                    <th>Weight</th>
                    <th>Weighted</th>
                    <th>Detail</th>
                </tr>
                {% for name, data in composite_score.breakdown.items() %}
                <tr>
                    <td style="text-align: left;">{{ name | replace('_', ' ') | title }}</td>
                    <td>{{ "%.0f"|format(data.score) }}</td>
                    <td>{{ "%.0f%%"|format(data.weight * 100) }}</td>
                    <td>{{ "%.1f"|format(data.weighted) }}</td>
                    <td style="text-align: left;">{{ data.detail }}</td>
                </tr>
                {% endfor %}
                <tr style="font-weight: bold; background: #e9ecef;">
                    <td style="text-align: left;">Total</td>
                    <td></td>
                    <td>100%</td>
                    <td>{{ "%.1f"|format(composite_score.total_score) }}</td>
                    <td>Grade: {{ composite_score.grade }}</td>
                </tr>
            </table>
        </div>
        {% endif %}
        
        <!-- Headline Metrics -->
        <div class="section">
            <h2>Headline Metrics (Best Config: {{ best_config }})</h2>
            <div class="metric-card">
                <div class="value">{{ "%.2f"|format(headline.sharpe) }}</div>
                <div class="label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ "%.1f%%"|format(headline.ann_ret * 100) }}</div>
                <div class="label">Annual Return</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ "%.1f%%"|format(headline.max_dd * 100) }}</div>
                <div class="label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ "%.1f%%"|format(headline.turnover * 100) }}</div>
                <div class="label">Turnover</div>
            </div>
            {% if headline.coverage is defined %}
            <div class="metric-card">
                <div class="value">{{ "%.0f"|format(headline.coverage) }}</div>
                <div class="label">Avg Securities/Day</div>
            </div>
            {% endif %}
        </div>
        
        <!-- Suite Results Table -->
        <div class="section">
            <h2>Suite Results</h2>
            {{ summary_table | safe }}
        </div>
        
        <!-- Lag Sensitivity -->
        {% if lag_table %}
        <div class="section">
            <h2>Lag Sensitivity</h2>
            <p>How metrics change as execution lag increases (signal decay analysis):</p>
            {{ lag_table | safe }}
            {% if lag_decay_warning %}
            <p style="color: #dc3545; font-weight: bold;">Warning: {{ lag_decay_warning }}</p>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Robustness Analysis -->
        <div class="section">
            <h2>Robustness Analysis</h2>
            <div class="two-column">
                {% if cap_table %}
                <div class="subsection">
                    <h3>Performance by Market Cap</h3>
                    <p style="font-size: 11px; color: #666; margin-bottom: 8px;">All metrics use 252-day annualization for consistency across slices.</p>
                    {{ cap_table | safe }}
                </div>
                {% endif %}
                {% if year_table %}
                <div class="subsection">
                    <h3>Performance by Year</h3>
                    <p style="font-size: 11px; color: #666; margin-bottom: 8px;">All metrics use 252-day annualization for consistency across slices.</p>
                    {{ year_table | safe }}
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Baseline Comparison -->
        <div class="section">
            <h2>Baseline Comparison</h2>
            {{ baseline_table | safe }}
        </div>
        
        <!-- Correlations -->
        <div class="section">
            <h2>Signal Uniqueness</h2>
            {{ correlation_table | safe }}
        </div>
        
        <!-- Factor Exposures -->
        {% if factor_table %}
        <div class="section">
            <h2>Risk Factor Exposures</h2>
            <p>Correlation of signal to common risk factors (high |correlation| suggests factor overlap):</p>
            {{ factor_table | safe }}
        </div>
        {% endif %}
        
        <!-- Information Coefficient -->
        {% if ic_stats %}
        <div class="section">
            <h2>Information Coefficient (IC)</h2>
            <p>Daily cross-sectional Spearman correlation between signal and forward returns:</p>
            <div class="metric-card">
                <div class="value" style="color: {% if ic_stats.mean > 0.02 %}#28a745{% elif ic_stats.mean > 0 %}#17a2b8{% else %}#dc3545{% endif %};">
                    {{ "%.4f"|format(ic_stats.mean) }}
                </div>
                <div class="label">Mean IC</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color: {% if ic_stats.t_stat > 2 %}#28a745{% elif ic_stats.t_stat > 1 %}#17a2b8{% else %}#dc3545{% endif %};">
                    {{ "%.2f"|format(ic_stats.t_stat) }}
                </div>
                <div class="label">t-Statistic</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ "%.1f%%"|format(ic_stats.hit_rate) }}</div>
                <div class="label">Hit Rate</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ "%.2f"|format(ic_stats.ir) }}</div>
                <div class="label">Info Ratio</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ ic_stats.n_days }}</div>
                <div class="label">Days</div>
            </div>
        </div>
        {% endif %}
        
        <!-- Coverage Metrics -->
        {% if coverage %}
        <div class="section">
            <h2>Signal Coverage</h2>
            <div class="metric-card">
                <div class="value">{{ "%.0f"|format(coverage.avg_securities_per_day) }}</div>
                <div class="label">Avg Securities/Day</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ coverage.unique_securities }}</div>
                <div class="label">Unique Securities</div>
            </div>
            <div class="metric-card">
                <div class="value">{{ coverage.total_days }}</div>
                <div class="label">Days with Signal</div>
            </div>
            {% if coverage.coverage_pct is defined and coverage.coverage_pct == coverage.coverage_pct %}
            <div class="metric-card">
                <div class="value">{{ "%.1f%%"|format(coverage.coverage_pct) }}</div>
                <div class="label">Universe Coverage</div>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def compute_verdict_from_summary(
    summary_df: pd.DataFrame,
    correlations: Optional[pd.DataFrame] = None,
    factor_exposures: Optional[pd.DataFrame] = None,
    cap_breakdown: Optional[pd.DataFrame] = None,
    year_breakdown: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute verdict from summary DataFrame (for use with cached artifacts).
    
    Returns:
        {'color': 'green'/'yellow'/'red', 'reasons': [...]}
    """
    reasons = []
    
    # Filter to signal configs only
    if 'type' in summary_df.columns:
        signal_summary = summary_df[summary_df['type'] == 'signal']
    else:
        signal_summary = summary_df
    
    if len(signal_summary) == 0:
        return {'color': 'red', 'reasons': ['No valid backtest results']}
    
    # Get best config by Sharpe
    best_row = signal_summary.loc[signal_summary['sharpe'].idxmax()]
    sharpe = best_row['sharpe']
    turnover = best_row.get('turnover', 0)
    max_dd = best_row.get('max_dd', 0)
    
    if np.isnan(sharpe):
        return {'color': 'red', 'reasons': ['Best config has NaN Sharpe']}
    
    # Check Sharpe decay with lag
    sharpe_decay = False
    lag0_row = signal_summary[signal_summary['config'] == 'lag0_residoff']
    if len(lag0_row) > 0:
        s0 = lag0_row['sharpe'].iloc[0]
        if not np.isnan(s0) and s0 > 0:
            for lag in [1, 2, 3, 5]:
                lag_key = f'lag{lag}_residoff'
                lag_row = signal_summary[signal_summary['config'] == lag_key]
                if len(lag_row) > 0:
                    s_lag = lag_row['sharpe'].iloc[0]
                    if not np.isnan(s_lag):
                        retention = s_lag / s0
                        thresholds = {1: 0.7, 2: 0.6, 3: 0.5, 5: 0.4}
                        threshold = thresholds.get(lag, 0.5)
                        if retention < threshold:
                            sharpe_decay = True
                            reasons.append(f"Sharpe decays with lag (lag0: {s0:.2f} -> lag{lag}: {s_lag:.2f}, {retention*100:.0f}% retained)")
                            break
    
    # Check residualization impact
    resid_sensitive = False
    off_row = signal_summary[signal_summary['config'] == 'lag0_residoff']
    on_row = signal_summary[signal_summary['config'] == 'lag0_residindustry']
    if len(off_row) > 0 and len(on_row) > 0:
        s_off = off_row['sharpe'].iloc[0]
        s_on = on_row['sharpe'].iloc[0]
        if not np.isnan(s_off) and not np.isnan(s_on) and s_off > 0 and (s_on / s_off) < 0.5:
            resid_sensitive = True
            reasons.append(f"Signal doesn't survive industry neutralization ({s_off:.2f} -> {s_on:.2f})")
    
    # Check baseline correlation
    high_baseline_corr = False
    if correlations is not None and len(correlations) > 0:
        for _, row in correlations.iterrows():
            if abs(row.get('signal_corr', 0)) > 0.5:
                high_baseline_corr = True
                reasons.append(f"High correlation with {row.get('baseline', 'unknown')}: {row['signal_corr']:.2f}")
    
    # Check factor exposures
    high_factor_exposure = False
    if factor_exposures is not None and len(factor_exposures) > 0:
        for _, row in factor_exposures.iterrows():
            if abs(row.get('correlation', 0)) > 0.3:
                high_factor_exposure = True
                reasons.append(f"High {row.get('factor', 'unknown')} factor exposure: {row['correlation']:.2f}")
    
    # Check cap breakdown
    small_cap_driven = False
    if cap_breakdown is not None and len(cap_breakdown) > 0 and 'Sharpe' in cap_breakdown.columns:
        cap_col = 'Cap Tier' if 'Cap Tier' in cap_breakdown.columns else cap_breakdown.columns[0]
        cap_sharpes = cap_breakdown.set_index(cap_col)['Sharpe']
        small_sharpe = cap_sharpes.get('Small Cap', np.nan)
        large_sharpe = cap_sharpes.get('Large Cap', np.nan)
        if not np.isnan(small_sharpe) and not np.isnan(large_sharpe):
            if large_sharpe > 0 and small_sharpe / large_sharpe > 2.0:
                small_cap_driven = True
                reasons.append(f"Driven by small caps (Small: {small_sharpe:.2f} vs Large: {large_sharpe:.2f})")
            elif large_sharpe <= 0 and small_sharpe > 0.5:
                small_cap_driven = True
                reasons.append(f"Only works in small caps (Small: {small_sharpe:.2f} vs Large: {large_sharpe:.2f})")
    
    # Check year breakdown
    year_inconsistent = False
    if year_breakdown is not None and len(year_breakdown) >= 2 and 'Sharpe' in year_breakdown.columns:
        year_sharpes = year_breakdown['Sharpe'].dropna()
        if len(year_sharpes) >= 2:
            neg_years = (year_sharpes < 0).sum()
            total_years = len(year_sharpes)
            if neg_years >= total_years / 2:
                year_inconsistent = True
                reasons.append(f"Inconsistent across years ({neg_years}/{total_years} years negative)")
    
    # Determine color
    if sharpe >= 1.0 and not sharpe_decay and not resid_sensitive and not high_baseline_corr and not small_cap_driven and not year_inconsistent:
        color = 'green'
        reasons.insert(0, f"Strong Sharpe ratio: {sharpe:.2f}")
        reasons.append("Survives lag and residualization tests")
        reasons.append("Low correlation to standard baselines")
        reasons.append("Consistent across cap tiers and years")
    elif sharpe >= 0.5:
        color = 'yellow'
        if sharpe < 1.0:
            reasons.insert(0, f"Moderate Sharpe ratio: {sharpe:.2f}")
        if turnover > 2.0:
            reasons.append(f"High turnover: {turnover:.1%}")
        if max_dd < -0.3:
            reasons.append(f"Large max drawdown: {max_dd:.1%}")
    else:
        color = 'red'
        reasons.insert(0, f"Weak Sharpe ratio: {sharpe:.2f}")
    
    return {'color': color, 'reasons': reasons}


def compute_verdict(suite_result: SuiteResult) -> Dict:
    """
    Compute traffic-light verdict from suite results.
    
    Returns:
        {'color': 'green'/'yellow'/'red', 'reasons': [...]}
    """
    reasons = []
    
    # Get best config metrics
    best_key = get_best_config(suite_result, 'sharpe')
    if best_key is None or best_key not in suite_result.results:
        return {'color': 'red', 'reasons': ['No valid backtest results']}
    
    best = suite_result.results[best_key]
    sharpe = best.sharpe
    turnover = best.turnover
    max_dd = best.max_drawdown
    
    # Handle NaN metrics
    if np.isnan(sharpe):
        return {'color': 'red', 'reasons': ['Best config has NaN Sharpe - insufficient data']}
    
    # Check Sharpe decay with lag - check all available lags
    sharpe_decay = False
    lag0_key = 'lag0_residoff'
    if lag0_key in suite_result.results:
        s0 = suite_result.results[lag0_key].sharpe
        if not np.isnan(s0) and s0 > 0:
            # Check all higher lags for decay
            for lag in [1, 2, 3, 5]:
                lag_key = f'lag{lag}_residoff'
                if lag_key in suite_result.results:
                    s_lag = suite_result.results[lag_key].sharpe
                    if not np.isnan(s_lag):
                        retention = s_lag / s0
                        # Flag if decay exceeds threshold for that lag
                        # lag1: 70%, lag2: 60%, lag3: 50%, lag5: 40%
                        thresholds = {1: 0.7, 2: 0.6, 3: 0.5, 5: 0.4}
                        threshold = thresholds.get(lag, 0.5)
                        if retention < threshold:
                            sharpe_decay = True
                            reasons.append(f"Sharpe decays with lag (lag0: {s0:.2f} -> lag{lag}: {s_lag:.2f}, {retention*100:.0f}% retained)")
                            break  # Only report the first significant decay
    
    # Check residualization impact
    resid_sensitive = False
    if 'lag0_residoff' in suite_result.results and 'lag0_residindustry' in suite_result.results:
        s_off = suite_result.results['lag0_residoff'].sharpe
        s_on = suite_result.results['lag0_residindustry'].sharpe
        if not np.isnan(s_off) and not np.isnan(s_on) and s_off > 0 and (s_on / s_off) < 0.5:
            resid_sensitive = True
            reasons.append(f"Signal doesn't survive industry neutralization ({s_off:.2f} -> {s_on:.2f})")
    
    # Check baseline correlation
    high_baseline_corr = False
    for _, row in suite_result.correlations.iterrows():
        if abs(row['signal_corr']) > 0.5:
            high_baseline_corr = True
            reasons.append(f"High correlation with {row['baseline']}: {row['signal_corr']:.2f}")
    
    # Check risk factor exposures
    high_factor_exposure = False
    if suite_result.factor_exposures is not None and len(suite_result.factor_exposures) > 0:
        for _, row in suite_result.factor_exposures.iterrows():
            if abs(row['correlation']) > 0.3:
                high_factor_exposure = True
                reasons.append(f"High {row['factor']} factor exposure: {row['correlation']:.2f}")
    
    # Check if driven by small caps
    small_cap_driven = False
    cap_data = _extract_cap_breakdown(suite_result, best_key)
    if len(cap_data) > 0 and 'Sharpe' in cap_data.columns:
        cap_sharpes = cap_data.set_index('Cap Tier')['Sharpe']
        if 'Small Cap' in cap_sharpes.index and 'Large Cap' in cap_sharpes.index:
            small_sharpe = cap_sharpes.get('Small Cap', np.nan)
            large_sharpe = cap_sharpes.get('Large Cap', np.nan)
            if not np.isnan(small_sharpe) and not np.isnan(large_sharpe):
                # Flag if small cap Sharpe is >2x large cap Sharpe
                if large_sharpe > 0 and small_sharpe / large_sharpe > 2.0:
                    small_cap_driven = True
                    reasons.append(f"Driven by small caps (Small: {small_sharpe:.2f} vs Large: {large_sharpe:.2f})")
                elif large_sharpe <= 0 and small_sharpe > 0.5:
                    small_cap_driven = True
                    reasons.append(f"Only works in small caps (Small: {small_sharpe:.2f} vs Large: {large_sharpe:.2f})")
    
    # Check year-over-year consistency
    year_inconsistent = False
    year_data = _extract_year_breakdown(suite_result, best_key)
    if len(year_data) >= 2 and 'Sharpe' in year_data.columns:
        year_sharpes = year_data['Sharpe'].dropna()
        if len(year_sharpes) >= 2:
            # Check for high variance in year-over-year Sharpe
            # Flag if some years are very negative while overall is positive
            neg_years = (year_sharpes < 0).sum()
            total_years = len(year_sharpes)
            if neg_years >= total_years / 2:
                year_inconsistent = True
                reasons.append(f"Inconsistent across years ({neg_years}/{total_years} years negative)")
    
    # Determine color
    if sharpe >= 1.0 and not sharpe_decay and not resid_sensitive and not high_baseline_corr and not small_cap_driven and not year_inconsistent:
        color = 'green'
        reasons.insert(0, f"Strong Sharpe ratio: {sharpe:.2f}")
        reasons.append("Survives lag and residualization tests")
        reasons.append("Low correlation to standard baselines")
        reasons.append("Consistent across cap tiers and years")
    elif sharpe >= 0.5:
        color = 'yellow'
        if sharpe < 1.0:
            reasons.insert(0, f"Moderate Sharpe ratio: {sharpe:.2f}")
        if turnover > 2.0:
            reasons.append(f"High turnover: {turnover:.1%}")
        if max_dd < -0.3:
            reasons.append(f"Large max drawdown: {max_dd:.1%}")
    else:
        color = 'red'
        reasons.insert(0, f"Weak Sharpe ratio: {sharpe:.2f}")
    
    return {'color': color, 'reasons': reasons}


def compute_composite_score(suite_result: SuiteResult, weights: Dict = None) -> Dict:
    """
    Compute a composite quality score (0-100) from suite results.
    
    Weights can be customized. Default weights:
        - sharpe: 25%
        - lag_stability: 15%
        - resid_stability: 15%
        - baseline_uniqueness: 15%
        - cap_consistency: 10%
        - year_consistency: 10%
        - factor_uniqueness: 10%
    
    Returns:
        {
            'total_score': float (0-100),
            'breakdown': {'metric': {'score': float, 'weight': float, 'weighted': float}},
            'grade': str (A/B/C/D/F)
        }
    """
    if weights is None:
        weights = {
            'sharpe': 0.25,
            'lag_stability': 0.15,
            'resid_stability': 0.15,
            'baseline_uniqueness': 0.15,
            'cap_consistency': 0.10,
            'year_consistency': 0.10,
            'factor_uniqueness': 0.10,
        }
    
    breakdown = {}
    
    # Get best config
    best_key = get_best_config(suite_result, 'sharpe')
    if best_key is None or best_key not in suite_result.results:
        return {'total_score': 0, 'breakdown': {}, 'grade': 'F'}
    
    best = suite_result.results[best_key]
    sharpe = best.sharpe if not np.isnan(best.sharpe) else 0
    
    # 1. Sharpe score (0-100): Scale from 0 to 2.0
    sharpe_score = min(100, max(0, sharpe / 2.0 * 100))
    breakdown['sharpe'] = {
        'score': sharpe_score,
        'weight': weights['sharpe'],
        'weighted': sharpe_score * weights['sharpe'],
        'detail': f"Sharpe: {sharpe:.2f}"
    }
    
    # 2. Lag stability (100 if no decay, 0 if >50% decay)
    lag_score = 100
    if 'lag0_residoff' in suite_result.results and 'lag2_residoff' in suite_result.results:
        s0 = suite_result.results['lag0_residoff'].sharpe
        s2 = suite_result.results['lag2_residoff'].sharpe
        if not np.isnan(s0) and not np.isnan(s2) and s0 > 0:
            retention = s2 / s0
            lag_score = min(100, max(0, retention * 100))
    breakdown['lag_stability'] = {
        'score': lag_score,
        'weight': weights['lag_stability'],
        'weighted': lag_score * weights['lag_stability'],
        'detail': f"Lag retention: {lag_score:.0f}%"
    }
    
    # 3. Residualization stability (100 if survives, 0 if doesn't)
    resid_score = 100
    if 'lag0_residoff' in suite_result.results and 'lag0_residindustry' in suite_result.results:
        s_off = suite_result.results['lag0_residoff'].sharpe
        s_on = suite_result.results['lag0_residindustry'].sharpe
        if not np.isnan(s_off) and not np.isnan(s_on) and s_off > 0:
            retention = s_on / s_off
            resid_score = min(100, max(0, retention * 100))
    breakdown['resid_stability'] = {
        'score': resid_score,
        'weight': weights['resid_stability'],
        'weighted': resid_score * weights['resid_stability'],
        'detail': f"Resid retention: {resid_score:.0f}%"
    }
    
    # 4. Baseline uniqueness (100 if no correlation, 0 if >0.5)
    baseline_score = 100
    if len(suite_result.correlations) > 0:
        max_corr = suite_result.correlations['signal_corr'].abs().max()
        if not np.isnan(max_corr):
            # Score: 100 if corr=0, 0 if corr>=0.5
            baseline_score = max(0, 100 - (max_corr / 0.5 * 100))
    breakdown['baseline_uniqueness'] = {
        'score': baseline_score,
        'weight': weights['baseline_uniqueness'],
        'weighted': baseline_score * weights['baseline_uniqueness'],
        'detail': f"Baseline uniqueness: {baseline_score:.0f}%"
    }
    
    # 5. Cap consistency (check if small cap isn't dominating)
    cap_score = 100
    cap_data = _extract_cap_breakdown(suite_result, best_key)
    if len(cap_data) > 0 and 'Sharpe' in cap_data.columns:
        cap_sharpes = cap_data.set_index('Cap Tier')['Sharpe']
        if 'Small Cap' in cap_sharpes.index and 'Large Cap' in cap_sharpes.index:
            small_s = cap_sharpes.get('Small Cap', 0)
            large_s = cap_sharpes.get('Large Cap', 0)
            if not np.isnan(small_s) and not np.isnan(large_s):
                if small_s > 0 and large_s > 0:
                    # Good if large cap is at least 50% of small cap
                    ratio = large_s / small_s
                    cap_score = min(100, ratio * 100)
                elif large_s <= 0 and small_s > 0:
                    cap_score = 0  # Only works in small caps
    breakdown['cap_consistency'] = {
        'score': cap_score,
        'weight': weights['cap_consistency'],
        'weighted': cap_score * weights['cap_consistency'],
        'detail': f"Cap consistency: {cap_score:.0f}%"
    }
    
    # 6. Year consistency (check for negative years)
    year_score = 100
    year_data = _extract_year_breakdown(suite_result, best_key)
    if len(year_data) >= 2 and 'Sharpe' in year_data.columns:
        year_sharpes = year_data['Sharpe'].dropna()
        if len(year_sharpes) >= 2:
            pos_years = (year_sharpes > 0).sum()
            total_years = len(year_sharpes)
            year_score = (pos_years / total_years) * 100
    breakdown['year_consistency'] = {
        'score': year_score,
        'weight': weights['year_consistency'],
        'weighted': year_score * weights['year_consistency'],
        'detail': f"Year consistency: {year_score:.0f}%"
    }
    
    # 7. Factor uniqueness (100 if no factor exposure, 0 if >0.3)
    factor_score = 100
    if suite_result.factor_exposures is not None and len(suite_result.factor_exposures) > 0:
        max_factor_corr = suite_result.factor_exposures['abs_correlation'].max()
        if not np.isnan(max_factor_corr):
            # Score: 100 if corr=0, 0 if corr>=0.3
            factor_score = max(0, 100 - (max_factor_corr / 0.3 * 100))
    breakdown['factor_uniqueness'] = {
        'score': factor_score,
        'weight': weights['factor_uniqueness'],
        'weighted': factor_score * weights['factor_uniqueness'],
        'detail': f"Factor uniqueness: {factor_score:.0f}%"
    }
    
    # Calculate total score
    total_score = sum(b['weighted'] for b in breakdown.values())
    
    # Determine grade
    if total_score >= 80:
        grade = 'A'
    elif total_score >= 65:
        grade = 'B'
    elif total_score >= 50:
        grade = 'C'
    elif total_score >= 35:
        grade = 'D'
    else:
        grade = 'F'
    
    return {
        'total_score': total_score,
        'breakdown': breakdown,
        'grade': grade,
    }


def _extract_cap_breakdown(suite_result: SuiteResult, config_key: str) -> pd.DataFrame:
    """Extract cap-tier breakdown from best config's summary."""
    if config_key not in suite_result.results:
        return pd.DataFrame()
    
    summary = suite_result.results[config_key].summary
    if summary is None or 'group' not in summary.columns:
        return pd.DataFrame()
    
    # Filter for cap groups (Large Cap, Medium Cap, Small Cap)
    cap_groups = ['Large Cap', 'Medium Cap', 'Small Cap']
    cap_data = summary[summary['group'].isin(cap_groups)].copy()
    
    if len(cap_data) == 0:
        return pd.DataFrame()
    
    # Select relevant columns
    display_cols = ['group', 'sharpe_ret', 'ret_ann', 'maxdraw', 'turnover']
    available_cols = [c for c in display_cols if c in cap_data.columns]
    cap_data = cap_data[available_cols].copy()
    
    # Rename for display
    col_rename = {
        'group': 'Cap Tier',
        'sharpe_ret': 'Sharpe',
        'ret_ann': 'Ann Return',
        'maxdraw': 'Max DD',
        'turnover': 'Turnover'
    }
    cap_data = cap_data.rename(columns={k: v for k, v in col_rename.items() if k in cap_data.columns})
    
    return cap_data


def _extract_year_breakdown(suite_result: SuiteResult, config_key: str) -> pd.DataFrame:
    """Extract year-by-year breakdown from best config's summary."""
    if config_key not in suite_result.results:
        return pd.DataFrame()
    
    summary = suite_result.results[config_key].summary
    if summary is None or 'group' not in summary.columns:
        return pd.DataFrame()
    
    # Filter for year groups (numeric or string years like '2017', '2018')
    # Exclude known non-year groups
    exclude_groups = ['overall', 'Large Cap', 'Medium Cap', 'Small Cap']
    year_data = summary[~summary['group'].isin(exclude_groups)].copy()
    
    # Try to identify year rows (4-digit numeric values)
    def is_year(val):
        try:
            year = int(str(val))
            return 2000 <= year <= 2030
        except (ValueError, TypeError):
            return False
    
    year_data = year_data[year_data['group'].apply(is_year)].copy()
    
    if len(year_data) == 0:
        return pd.DataFrame()
    
    # Select relevant columns
    display_cols = ['group', 'sharpe_ret', 'ret_ann', 'maxdraw', 'num_date']
    available_cols = [c for c in display_cols if c in year_data.columns]
    year_data = year_data[available_cols].copy()
    
    # Rename for display
    col_rename = {
        'group': 'Year',
        'sharpe_ret': 'Sharpe',
        'ret_ann': 'Ann Return',
        'maxdraw': 'Max DD',
        'num_date': 'Trading Days'
    }
    year_data = year_data.rename(columns={k: v for k, v in col_rename.items() if k in year_data.columns})
    
    # Sort by year
    year_data = year_data.sort_values('Year')
    
    return year_data


def generate_tearsheet(
    suite_result: SuiteResult,
    signal_name: str,
    catalog: dict,
    output_path: str = 'tearsheet.html',
) -> str:
    """
    Generate HTML tear sheet from suite results.
    
    Returns:
        Path to generated HTML file
    """
    # Compute verdict
    verdict = compute_verdict(suite_result)
    
    # Compute composite score
    composite_score = compute_composite_score(suite_result)
    
    # Get best config
    best_config = get_best_config(suite_result, 'sharpe') or 'N/A'
    
    # Headline metrics
    headline = {'sharpe': 0, 'ann_ret': 0, 'max_dd': 0, 'turnover': 0}
    if best_config in suite_result.results:
        best = suite_result.results[best_config]
        headline = {
            'sharpe': best.sharpe,
            'ann_ret': best.annual_return,
            'max_dd': best.max_drawdown,
            'turnover': best.turnover,
        }
    
    # Extract cap and year breakdown from best config
    cap_data = _extract_cap_breakdown(suite_result, best_config)
    year_data = _extract_year_breakdown(suite_result, best_config)
    
    # Extract lag sensitivity table
    lag_data = get_lag_sensitivity_table(suite_result)
    lag_table = None
    lag_decay_warning = None
    if len(lag_data) > 0:
        # Check for lag decay warning
        lag0_sharpe = lag_data[lag_data['Lag'] == 0]['Sharpe'].max() if 0 in lag_data['Lag'].values else None
        if lag0_sharpe and lag0_sharpe > 0:
            max_lag = lag_data['Lag'].max()
            max_lag_sharpe = lag_data[lag_data['Lag'] == max_lag]['Sharpe'].max()
            if max_lag_sharpe < lag0_sharpe * 0.5:
                lag_decay_warning = f"Signal decays >50% from lag-0 ({lag0_sharpe:.2f}) to lag-{max_lag} ({max_lag_sharpe:.2f})"
        
        lag_table = lag_data.to_html(index=False, float_format='%.3f')
    
    # Format tables
    signal_summary = suite_result.summary[suite_result.summary['type'] == 'signal']
    baseline_summary = suite_result.summary[suite_result.summary['type'] == 'baseline']
    
    # Format factor exposures table
    factor_table = None
    if suite_result.factor_exposures is not None and len(suite_result.factor_exposures) > 0:
        factor_df = suite_result.factor_exposures[['factor', 'correlation']].copy()
        factor_df.columns = ['Factor', 'Correlation']
        factor_table = factor_df.to_html(index=False, float_format='%.3f')
    
    # Render template
    template = Template(TEARSHEET_TEMPLATE)
    html = template.render(
        signal_name=signal_name,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        snapshot_id=catalog.get('snapshot_id', 'unknown'),
        verdict=verdict,
        composite_score=composite_score,
        best_config=best_config,
        headline=headline,
        summary_table=signal_summary.to_html(index=False, float_format='%.3f'),
        baseline_table=baseline_summary.to_html(index=False, float_format='%.3f'),
        correlation_table=suite_result.correlations.to_html(index=False, float_format='%.3f'),
        cap_table=cap_data.to_html(index=False, float_format='%.3f') if len(cap_data) > 0 else None,
        year_table=year_data.to_html(index=False, float_format='%.3f') if len(year_data) > 0 else None,
        lag_table=lag_table,
        lag_decay_warning=lag_decay_warning,
        factor_table=factor_table,
        coverage=suite_result.coverage,
        ic_stats=suite_result.ic_stats,
    )
    
    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    
    print(f"Tear sheet saved to: {output}")
    return str(output)
