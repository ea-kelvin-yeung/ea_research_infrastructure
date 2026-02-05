"""
Streamlit UI: Simple launcher and viewer for backtest suite.
~200 lines - single-page app.

Run with: streamlit run poc/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from poc.catalog import load_catalog, list_snapshots, create_snapshot
from poc.suite import run_suite, DEFAULT_GRID
from poc.tearsheet import generate_tearsheet
from poc.tracking import log_run, get_run_history


st.set_page_config(page_title="Backtest Suite Runner", page_icon="📊", layout="wide")


def filter_catalog(catalog: dict, start_date: str, end_date: str) -> dict:
    """Filter catalog data to date range for faster testing."""
    filtered = catalog.copy()
    filtered['ret'] = catalog['ret'][
        (catalog['ret']['date'] >= start_date) & 
        (catalog['ret']['date'] <= end_date)
    ].copy()
    filtered['risk'] = catalog['risk'][
        (catalog['risk']['date'] >= start_date) & 
        (catalog['risk']['date'] <= end_date)
    ].copy()
    return filtered


def main():
    st.title("📊 Backtest Suite Runner")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Snapshot selection
        snapshots = list_snapshots('snapshots')
        if not snapshots:
            st.warning("No snapshots found. Create one first.")
            if st.button("Create Snapshot from data/"):
                try:
                    create_snapshot('data', 'snapshots')
                    st.success("Snapshot created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
            return
        
        snapshot = st.selectbox("Snapshot", snapshots)
        
        # Suite configuration
        st.subheader("Suite Options")
        lags = st.multiselect("Lags", [0, 1, 2, 3, 5], default=[0, 1])
        resid_opts = st.multiselect("Residualize", ['off', 'industry', 'all'], default=['off'])
        include_baselines = st.checkbox("Include Baselines", value=False)
        log_to_mlflow = st.checkbox("Log to MLflow", value=True)
        
        # Date range for faster testing
        st.subheader("Date Range")
        st.caption("Limit data for faster runs")
        start_date = st.date_input("Start", value=pd.Timestamp('2018-01-01'))
        end_date = st.date_input("End", value=pd.Timestamp('2018-06-30'))
    
    # Main area - tabs
    tab1, tab2, tab3 = st.tabs(["Run Suite", "Results", "History"])
    
    # Tab 1: Run Suite
    with tab1:
        st.header("Upload Signal")
        
        uploaded = st.file_uploader(
            "Drag & drop your signal file here", 
            type=['parquet', 'csv', 'pkl', 'pickle'],
            help="Supported formats: .parquet, .csv, .pkl. Must have columns: security_id, date_sig, date_avail, signal"
        )
        signal_name = st.text_input("Signal Name", value="my_signal")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            run_button = st.button("🚀 Run Suite", type="primary")
        
        if run_button and uploaded:
            with st.spinner("Running backtest suite..."):
                try:
                    # Load signal
                    if uploaded.name.endswith('.parquet'):
                        signal_df = pd.read_parquet(uploaded)
                    elif uploaded.name.endswith('.pkl') or uploaded.name.endswith('.pickle'):
                        import pickle
                        signal_df = pickle.load(uploaded)
                    else:
                        signal_df = pd.read_csv(uploaded, parse_dates=['date_sig', 'date_avail'])
                    
                    # Filter signal to date range
                    start_str = str(start_date)
                    end_str = str(end_date)
                    signal_df['date_sig'] = pd.to_datetime(signal_df['date_sig'])
                    signal_df = signal_df[
                        (signal_df['date_sig'] >= start_str) & 
                        (signal_df['date_sig'] <= end_str)
                    ]
                    st.info(f"Signal: {len(signal_df):,} rows ({start_str} to {end_str})")
                    
                    # Load and filter catalog
                    catalog = load_catalog(f"snapshots/{snapshot}")
                    catalog = filter_catalog(catalog, start_str, end_str)
                    st.info(f"Data: {len(catalog['ret']):,} returns, {len(catalog['risk']):,} risk rows")
                    
                    # Run suite
                    grid = {'lags': lags, 'residualize': resid_opts}
                    result = run_suite(
                        signal_df, catalog, grid=grid, 
                        include_baselines=include_baselines,
                        baseline_start_date=start_str,
                        baseline_end_date=end_str,
                    )
                    
                    # Store in session state
                    st.session_state['result'] = result
                    st.session_state['catalog'] = catalog
                    st.session_state['signal_name'] = signal_name
                    
                    # Generate tearsheet
                    tearsheet_path = generate_tearsheet(result, signal_name, catalog, f"artifacts/{signal_name}_tearsheet.html")
                    st.session_state['tearsheet_path'] = tearsheet_path
                    
                    # Log to MLflow
                    if log_to_mlflow:
                        run_id = log_run(result, signal_name, catalog, tearsheet_path)
                        st.session_state['run_id'] = run_id
                    
                    st.success("Suite completed! Check the Results tab.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        elif run_button and not uploaded:
            st.warning("Please upload a signal file first.")
    
    # Tab 2: Results
    with tab2:
        if 'result' not in st.session_state:
            st.info("Run a suite first to see results.")
        else:
            result = st.session_state['result']
            signal_name = st.session_state.get('signal_name', 'signal')
            
            st.header(f"Results: {signal_name}")
            
            # Summary table
            st.subheader("Suite Summary")
            st.dataframe(
                result.summary.style.format({
                    'sharpe': '{:.2f}',
                    'ann_ret': '{:.2%}',
                    'max_dd': '{:.2%}',
                    'turnover': '{:.2%}',
                }),
                width='stretch'
            )
            
            # Cumulative return plot
            st.subheader("Cumulative Returns")
            
            # Collect daily series from all configs
            daily_data = []
            for key, res in result.results.items():
                if 'cumret' in res.daily.columns:
                    df = res.daily[['date', 'cumret']].copy()
                    df['config'] = key
                    daily_data.append(df)
            
            if daily_data:
                combined = pd.concat(daily_data)
                fig = px.line(combined, x='date', y='cumret', color='config', 
                             title='Cumulative Return by Config')
                st.plotly_chart(fig, width='stretch')
            
            # Correlations
            st.subheader("Baseline Correlations")
            st.dataframe(result.correlations.style.format({
                'signal_corr': '{:.3f}',
                'pnl_corr': '{:.3f}',
            }))
            
            # Tearsheet link
            if 'tearsheet_path' in st.session_state:
                st.subheader("Tear Sheet")
                tearsheet_path = st.session_state['tearsheet_path']
                st.markdown(f"[Open Tear Sheet]({tearsheet_path})")
                
                with open(tearsheet_path) as f:
                    st.components.v1.html(f.read(), height=800, scrolling=True)
    
    # Tab 3: History
    with tab3:
        st.header("Run History")
        
        if st.button("Refresh"):
            st.rerun()
        
        runs = get_run_history()
        
        if not runs:
            st.info("No runs found in MLflow. Run a suite first.")
        else:
            runs_df = pd.DataFrame(runs)
            
            # Display key columns
            display_cols = ['run_id', 'start_time', 'tags.signal_name', 'tags.snapshot_id']
            available_cols = [c for c in display_cols if c in runs_df.columns]
            
            if available_cols:
                st.dataframe(runs_df[available_cols], width='stretch')
            else:
                st.dataframe(runs_df, width='stretch')
            
            st.markdown("[Open MLflow UI](http://localhost:5000)")


if __name__ == "__main__":
    main()
