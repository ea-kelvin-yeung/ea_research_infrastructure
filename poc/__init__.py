"""
Backtest PoC Infrastructure

Modules:
    contract    - Signal validation and date alignment
    catalog     - Data snapshot management
    wrapper     - Backtest wrapper with clean API
    baselines   - Standard comparison signals
    suite       - Suite runner for config matrix
    tracking    - MLflow logging
    tearsheet   - HTML report generation

Quick start:
    from poc.catalog import load_catalog
    from poc.suite import run_suite
    from poc.tearsheet import generate_tearsheet
    
    catalog = load_catalog('snapshots/default')
    result = run_suite(signal_df, catalog)
    generate_tearsheet(result, 'my_signal', catalog)
"""

from . import contract
from . import catalog
from . import wrapper
from . import baselines
from . import suite
from . import tracking
from . import tearsheet
