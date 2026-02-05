"""
Tear Sheet Generator: Create HTML reports from suite results.
~200 lines - includes verdict logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from jinja2 import Template

from .suite import SuiteResult, get_best_config


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
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ signal_name }}</h1>
        <p class="timestamp">Generated: {{ timestamp }} | Snapshot: {{ snapshot_id }}</p>
        
        <!-- Verdict Panel -->
        <div class="verdict {{ verdict.color }}">
            <h3>Verdict: {{ verdict.color | upper }}</h3>
            <ul>
            {% for reason in verdict.reasons %}
                <li>{{ reason }}</li>
            {% endfor %}
            </ul>
        </div>
        
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
        </div>
        
        <!-- Suite Results Table -->
        <div class="section">
            <h2>Suite Results</h2>
            {{ summary_table | safe }}
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
        
        <!-- Robustness by Year -->
        {% if year_table %}
        <div class="section">
            <h2>Robustness by Year</h2>
            {{ year_table | safe }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


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
    
    # Check Sharpe decay with lag
    lag0_key = 'lag0_residoff'
    lag1_key = 'lag1_residoff'
    lag2_key = 'lag2_residoff'
    
    sharpe_decay = False
    if lag0_key in suite_result.results and lag2_key in suite_result.results:
        s0 = suite_result.results[lag0_key].sharpe
        s2 = suite_result.results[lag2_key].sharpe
        if s0 > 0 and (s2 / s0) < 0.5:
            sharpe_decay = True
            reasons.append(f"Sharpe decays significantly with lag (lag0: {s0:.2f} -> lag2: {s2:.2f})")
    
    # Check residualization impact
    resid_sensitive = False
    if 'lag0_residoff' in suite_result.results and 'lag0_residindustry' in suite_result.results:
        s_off = suite_result.results['lag0_residoff'].sharpe
        s_on = suite_result.results['lag0_residindustry'].sharpe
        if s_off > 0 and (s_on / s_off) < 0.5:
            resid_sensitive = True
            reasons.append(f"Signal doesn't survive industry neutralization ({s_off:.2f} -> {s_on:.2f})")
    
    # Check baseline correlation
    high_baseline_corr = False
    for _, row in suite_result.correlations.iterrows():
        if abs(row['signal_corr']) > 0.5:
            high_baseline_corr = True
            reasons.append(f"High correlation with {row['baseline']}: {row['signal_corr']:.2f}")
    
    # Determine color
    if sharpe >= 1.0 and not sharpe_decay and not resid_sensitive and not high_baseline_corr:
        color = 'green'
        reasons.insert(0, f"Strong Sharpe ratio: {sharpe:.2f}")
        reasons.append("Survives lag and residualization tests")
        reasons.append("Low correlation to standard baselines")
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
    
    # Format tables
    signal_summary = suite_result.summary[suite_result.summary['type'] == 'signal']
    baseline_summary = suite_result.summary[suite_result.summary['type'] == 'baseline']
    
    # Render template
    template = Template(TEARSHEET_TEMPLATE)
    html = template.render(
        signal_name=signal_name,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        snapshot_id=catalog.get('snapshot_id', 'unknown'),
        verdict=verdict,
        best_config=best_config,
        headline=headline,
        summary_table=signal_summary.to_html(index=False, float_format='%.3f'),
        baseline_table=baseline_summary.to_html(index=False, float_format='%.3f'),
        correlation_table=suite_result.correlations.to_html(index=False, float_format='%.3f'),
        year_table=None,  # TODO: Add year breakdown
    )
    
    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    
    print(f"Tear sheet saved to: {output}")
    return str(output)
