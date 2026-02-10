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
from poc.tearsheet import generate_tearsheet, compute_verdict, compute_composite_score, _extract_cap_breakdown, _extract_year_breakdown
from poc.suite import get_best_config


@st.cache_resource(show_spinner="Loading data snapshot...")
def get_cached_catalog(snapshot_path: str):
    """Load catalog once and cache in memory across reruns."""
    return load_catalog(snapshot_path, use_master=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_run_history():
    """Get run history with 60-second cache to avoid repeated MLflow queries."""
    return get_run_history()
from poc.tracking import log_run, get_run_history, get_git_sha, compute_signal_hash, delete_runs
from poc.charts import (
    plot_lag_sensitivity, plot_lag_sensitivity_from_summary, plot_decile_returns, 
    plot_factor_exposure_bars, plot_coverage_over_time, plot_ic_over_time
)
from poc.compare import compare_runs, get_overlay_data, compute_cumret_diff, clear_run_cache


st.set_page_config(page_title="Backtest Suite Runner", page_icon="📊", layout="wide")

# Custom CSS to reduce metric font size
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


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
        
        # Data snapshot selection
        snapshots = list_snapshots('snapshots')
        if not snapshots:
            st.warning("No data snapshots found. Create one first.")
            if st.button("Create Data Snapshot from data/"):
                try:
                    create_snapshot('data', 'snapshots')
                    st.success("Data snapshot created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
            return
        
        selected_snapshot = st.selectbox("Data Snapshot", snapshots, help="Pre-built market data (returns, risk factors, dates)")
        
        # Cache controls
        with st.expander("Data Cache", expanded=False):
            st.caption("Data is cached in memory for faster reruns.")
            if st.button("Clear Data Cache", help="Clear cached snapshot data to reload from disk"):
                get_cached_catalog.clear()
                st.success("Data cache cleared!")
                st.rerun()
            if st.button("Clear Compare Cache", help="Clear cached MLflow run data"):
                clear_run_cache()
                get_cached_run_history.clear()
                st.success("Compare cache cleared!")
                st.rerun()
        
        # Suite configuration
        st.subheader("Suite Options")
        lags = st.multiselect("Lags", [0, 1, 2, 3, 5], default=[0, 1])
        resid_opts = st.multiselect(
            "Residualize", 
            ['off', 'industry', 'factor', 'all'], 
            default=['off'],
            help="off=raw signal, industry=demean by industry, factor=regress on risk factors, all=both"
        )
        include_baselines = st.checkbox("Include Baselines", value=False, help="Run baseline backtests (required for PnL correlation)")
        log_to_mlflow = st.checkbox("Log to MLflow", value=True)
        
        # Date range for faster testing
        st.subheader("Date Range")
        start_date = st.date_input("Start", value=pd.Timestamp('2018-01-01'))
        end_date = st.date_input("End", value=pd.Timestamp('2018-06-30'))
        
        # Universe filter
        st.subheader("Universe Filter")
        universe_filter = st.radio(
            "Filter by universe",
            ["All Securities", "Investable Universe", "Non-Investable Universe"],
            index=0,
            help="Filter signal to securities in/out of the investable trading universe"
        )
    
    # Main area - tabs
    tab1, tab2, tab3 = st.tabs(["Run Suite", "Compare", "History"])
    
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
            import time
            total_start = time.time()
            step_times = []
            
            # Create a status container for live updates
            status_container = st.container()
            progress_bar = st.progress(0)
            log_container = st.empty()
            
            def log_step(step_name: str, step_time: float, progress: float):
                step_times.append((step_name, step_time))
                progress_bar.progress(progress)
                log_text = "\n".join([f"✓ {name}: {t:.2f}s" for name, t in step_times])
                log_container.code(log_text, language=None)
            
            try:
                # Step 1: Load signal
                step_start = time.time()
                if uploaded.name.endswith('.parquet'):
                    signal_df = pd.read_parquet(uploaded)
                elif uploaded.name.endswith('.pkl') or uploaded.name.endswith('.pickle'):
                    import pickle
                    signal_df = pickle.load(uploaded)
                else:
                    signal_df = pd.read_csv(uploaded, parse_dates=['date_sig', 'date_avail'])
                log_step(f"Load signal ({len(signal_df):,} rows)", time.time() - step_start, 0.1)
                
                # Step 2: Filter signal to date range
                step_start = time.time()
                start_str = str(start_date)
                end_str = str(end_date)
                signal_df['date_sig'] = pd.to_datetime(signal_df['date_sig'])
                original_len = len(signal_df)
                signal_df = signal_df[
                    (signal_df['date_sig'] >= start_str) & 
                    (signal_df['date_sig'] <= end_str)
                ]
                log_step(f"Filter signal ({original_len:,} → {len(signal_df):,})", time.time() - step_start, 0.15)
                
                # Step 3: Load catalog (cached in memory after first load)
                step_start = time.time()
                catalog = get_cached_catalog(f"snapshots/{selected_snapshot}")
                catalog = filter_catalog(catalog, start_str, end_str)
                log_step(f"Load catalog ({len(catalog['ret']):,} ret rows)", time.time() - step_start, 0.25)
                
                # Step 4: Apply universe filter if requested
                if universe_filter != "All Securities":
                    step_start = time.time()
                    desc_path = Path('data/descriptor.parquet')
                    if desc_path.exists():
                        desc = pd.read_parquet(desc_path, columns=['security_id', 'as_of_date', 'universe_flag'])
                        desc['as_of_date'] = pd.to_datetime(desc['as_of_date'])
                        desc = desc[(desc['as_of_date'] >= start_str) & (desc['as_of_date'] <= end_str)]
                        
                        target_flag = 1 if universe_filter == "Investable Universe" else 0
                        universe_secs = desc[desc['universe_flag'] == target_flag][['security_id', 'as_of_date']].drop_duplicates()
                        
                        original_len = len(signal_df)
                        signal_df = signal_df.merge(
                            universe_secs.rename(columns={'as_of_date': 'date_sig'}),
                            on=['security_id', 'date_sig'],
                            how='inner'
                        )
                        log_step(f"Universe filter ({original_len:,} → {len(signal_df):,})", time.time() - step_start, 0.30)
                    else:
                        st.warning("descriptor.parquet not found, skipping universe filter")
                
                # Step 5: Run suite (main backtest)
                step_start = time.time()
                grid = {'lags': lags, 'residualize': resid_opts}
                num_configs = len(lags) * len(resid_opts)
                result = run_suite(
                    signal_df, catalog, grid=grid, 
                    include_baselines=include_baselines,
                    baseline_start_date=start_str,
                    baseline_end_date=end_str,
                )
                baselines_note = f" + {len(result.baselines)} baselines" if include_baselines else ""
                log_step(f"Run suite ({num_configs} configs{baselines_note})", time.time() - step_start, 0.75)
                
                # Store in session state
                st.session_state['result'] = result
                st.session_state['catalog'] = catalog
                st.session_state['signal_name'] = signal_name
                st.session_state['signal_df'] = signal_df
                st.session_state['snapshot_id'] = selected_snapshot
                
                # Step 6: Generate tearsheet
                step_start = time.time()
                tearsheet_path = generate_tearsheet(result, signal_name, catalog, f"artifacts/{signal_name}_tearsheet.html")
                st.session_state['tearsheet_path'] = tearsheet_path
                log_step("Generate tearsheet", time.time() - step_start, 0.90)
                
                # Step 7: Log to MLflow
                if log_to_mlflow:
                    step_start = time.time()
                    run_id = log_run(result, signal_name, catalog, tearsheet_path, signal_df=signal_df)
                    st.session_state['run_id'] = run_id
                    log_step("Log to MLflow", time.time() - step_start, 1.0)
                else:
                    progress_bar.progress(1.0)
                
                # Final summary
                total_time = time.time() - total_start
                st.success(f"Suite completed in {total_time:.1f}s! Check the History tab to view results.")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        elif run_button and not uploaded:
            st.warning("Please upload a signal file first.")
    
    # Tab 2: Compare
    with tab2:
        st.header("Compare Runs")
        
        runs = get_cached_run_history()
        
        if not runs or len(runs) < 2:
            st.info("Need at least 2 runs in MLflow to compare. Run some backtests first.")
        else:
            runs_df = pd.DataFrame(runs)
            
            # Create labels for run selection
            if 'tags.signal_name' in runs_df.columns:
                runs_df['label'] = runs_df.apply(
                    lambda r: f"{r.get('tags.signal_name', 'unknown')} ({r['run_id'][:8]}...) - {str(r.get('start_time', ''))[:16]}", 
                    axis=1
                )
            else:
                runs_df['label'] = runs_df['run_id']
            
            # Two column selectors for Run A and Run B
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Run A (Baseline)")
                run_a_label = st.selectbox("Select Run A", runs_df['label'].tolist(), key="compare_run_a")
                run_a_id = runs_df[runs_df['label'] == run_a_label].iloc[0]['run_id']
            
            with col2:
                st.subheader("Run B (New)")
                # Default to second run if available
                default_idx = min(1, len(runs_df) - 1)
                run_b_label = st.selectbox("Select Run B", runs_df['label'].tolist(), index=default_idx, key="compare_run_b")
                run_b_id = runs_df[runs_df['label'] == run_b_label].iloc[0]['run_id']
            
            if run_a_id == run_b_id:
                st.warning("Please select two different runs to compare.")
            else:
                if st.button("Compare Runs", type="primary"):
                    import time
                    compare_start = time.time()
                    
                    try:
                        with st.spinner(f"Loading runs from MLflow..."):
                            result = compare_runs(run_a_id, run_b_id)
                            st.session_state['compare_result'] = result
                        
                        compare_time = time.time() - compare_start
                        st.success(f"Comparison loaded in {compare_time:.2f}s")
                    except Exception as e:
                        st.error(f"Error comparing runs: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # Display comparison results
                if 'compare_result' in st.session_state:
                    result = st.session_state['compare_result']
                    
                    # Run info summary
                    st.subheader("Run Information")
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.markdown("**Run A**")
                        st.write(f"Signal: {result.run_a.get('signal_name', 'N/A')}")
                        st.write(f"Snapshot: {result.run_a.get('snapshot_id', 'N/A')}")
                        st.write(f"Git SHA: {result.run_a.get('git_sha', 'N/A')}")
                    with info_col2:
                        st.markdown("**Run B**")
                        st.write(f"Signal: {result.run_b.get('signal_name', 'N/A')}")
                        st.write(f"Snapshot: {result.run_b.get('snapshot_id', 'N/A')}")
                        st.write(f"Git SHA: {result.run_b.get('git_sha', 'N/A')}")
                    
                    # Metrics comparison
                    st.subheader("Metrics Comparison")
                    if len(result.metrics_diff) > 0:
                        # Style the diff table
                        def highlight_better(row):
                            if row['Better'] == 'B':
                                return ['', '', '', 'background-color: #d4edda', 'background-color: #d4edda', 'color: green']
                            elif row['Better'] == 'A':
                                return ['', '', '', 'background-color: #f8d7da', 'background-color: #f8d7da', 'color: red']
                            else:
                                return [''] * 6
                        
                        styled = result.metrics_diff.style.apply(highlight_better, axis=1).format({
                            'Run A': '{:.4f}',
                            'Run B': '{:.4f}',
                            'Diff': '{:+.4f}',
                            'Diff %': '{:+.1f}%',
                        }, na_rep='N/A')
                        st.dataframe(styled, width='stretch')
                    else:
                        st.info("No comparable metrics found.")
                    
                    # Overlay plots
                    st.subheader("Cumulative Return Comparison")
                    if result.daily_a is not None or result.daily_b is not None:
                        overlay_data = get_overlay_data(
                            result.daily_a, result.daily_b,
                            label_a=result.run_a.get('signal_name', 'Run A'),
                            label_b=result.run_b.get('signal_name', 'Run B')
                        )
                        
                        if len(overlay_data) > 0:
                            fig = px.line(
                                overlay_data, x='date', y='cumret', color='run',
                                title='Cumulative Return: Run A vs Run B'
                            )
                            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
                            st.plotly_chart(fig, width='stretch', key="compare_cumret")
                            
                            # Difference plot
                            diff_data = compute_cumret_diff(result.daily_a, result.daily_b)
                            if len(diff_data) > 0:
                                st.subheader("Return Difference (Run B - Run A)")
                                fig_diff = go.Figure()
                                fig_diff.add_trace(go.Scatter(
                                    x=diff_data['date'],
                                    y=diff_data['cumret_diff'] * 100,
                                    fill='tozeroy',
                                    name='Difference',
                                    line=dict(color='steelblue'),
                                ))
                                fig_diff.add_hline(y=0, line_dash="solid", line_color="gray")
                                fig_diff.update_layout(
                                    yaxis_title="Cum Return Diff (%)",
                                    xaxis_title="Date",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_diff, width='stretch', key="compare_diff")
                        else:
                            st.info("No daily data available for overlay plot.")
                    else:
                        st.info("Daily data not available for one or both runs.")
    
    # Tab 3: History
    with tab3:
        st.header("Past Experiments")
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("Refresh", key="history_refresh"):
                # Clear the run history cache and reload
                get_cached_run_history.clear()
                st.rerun()
        
        runs = get_cached_run_history()
        
        if not runs:
            st.info("No runs found in MLflow. Run a suite first.")
        else:
            runs_df = pd.DataFrame(runs)
            
            # Create selection options
            if 'tags.signal_name' in runs_df.columns:
                runs_df['label'] = runs_df.apply(
                    lambda r: f"{r.get('tags.signal_name', 'unknown')} - {str(r.get('start_time', ''))[:16]}", 
                    axis=1
                )
            else:
                runs_df['label'] = runs_df['run_id'].str[:12] + '...'
            
            # === Run Management Section ===
            with st.expander("Manage Runs", expanded=False):
                st.caption("Select runs to delete")
                
                # Get all run IDs for checkbox key management
                all_run_ids = runs_df['run_id'].tolist()
                
                # Select All / Deselect All buttons
                mgmt_col1, mgmt_col2, mgmt_col3 = st.columns([1, 1, 2])
                with mgmt_col1:
                    if st.button("Select All", key="select_all_runs"):
                        # Set all checkbox states to True
                        for rid in all_run_ids:
                            st.session_state[f"run_select_{rid}"] = True
                        st.rerun()
                with mgmt_col2:
                    if st.button("Deselect All", key="deselect_all_runs"):
                        # Set all checkbox states to False
                        for rid in all_run_ids:
                            st.session_state[f"run_select_{rid}"] = False
                        st.rerun()
                
                # Display runs as checkboxes
                selected_run_ids = []
                for idx, row in runs_df.iterrows():
                    run_id = row['run_id']
                    label = row['label']
                    sharpe = row.get('metrics.best_sharpe', None)
                    sharpe_str = f" (Sharpe: {sharpe:.2f})" if sharpe is not None else ""
                    
                    checkbox_key = f"run_select_{run_id}"
                    if st.checkbox(f"{label}{sharpe_str}", key=checkbox_key):
                        selected_run_ids.append(run_id)
                
                # Delete button
                num_selected = len(selected_run_ids)
                if num_selected > 0:
                    st.warning(f"{num_selected} run(s) selected for deletion")
                    if st.button(f"Delete {num_selected} Selected Run(s)", type="primary", key="delete_runs_btn"):
                        with st.spinner("Deleting runs..."):
                            result = delete_runs(selected_run_ids)
                            if result['deleted']:
                                st.success(f"Deleted {len(result['deleted'])} run(s)")
                            if result['failed']:
                                for fail in result['failed']:
                                    st.error(f"Failed to delete {fail['run_id'][:12]}: {fail['error']}")
                            # Clear checkbox states and refresh
                            for rid in selected_run_ids:
                                if f"run_select_{rid}" in st.session_state:
                                    del st.session_state[f"run_select_{rid}"]
                            get_cached_run_history.clear()
                            st.rerun()
            
            # === View Selected Run ===
            st.subheader("View Experiment")
            
            # Select a run to view
            selected_label = st.selectbox("Select an experiment to view", runs_df['label'].tolist())
            selected_run = runs_df[runs_df['label'] == selected_label].iloc[0]
            run_id = selected_run['run_id']
            
            # Load artifacts from MLflow
            try:
                import mlflow
                import json
                client = mlflow.tracking.MlflowClient()
                artifacts = client.list_artifacts(run_id)
                artifact_names = [a.path for a in artifacts]
                
                # Helper to load artifact
                def load_artifact(pattern, file_type='parquet'):
                    artifact = next((a for a in artifact_names if pattern in a and a.endswith(f'.{file_type}')), None)
                    if artifact:
                        local_path = client.download_artifacts(run_id, artifact)
                        if file_type == 'parquet':
                            return pd.read_parquet(local_path)
                        elif file_type == 'json':
                            with open(local_path) as f:
                                return json.load(f)
                        elif file_type == 'csv':
                            return pd.read_csv(local_path)
                    return None
                
                # Load all artifacts
                daily_df = load_artifact('daily', 'parquet')
                summary_df = load_artifact('summary', 'csv')
                ic_series = load_artifact('ic_series', 'parquet')
                ic_stats = load_artifact('ic_stats', 'json')
                factor_exposures = load_artifact('factor_exposures', 'parquet')
                correlations = load_artifact('correlations', 'parquet')
                coverage = load_artifact('coverage', 'json')
                verdict_data = load_artifact('verdict', 'json')
                composite_score_data = load_artifact('composite_score', 'json')
                cap_breakdown = load_artifact('cap_breakdown', 'csv')
                year_breakdown = load_artifact('year_breakdown', 'csv')
                
                # === Verdict Panel (at top) ===
                if verdict_data:
                    verdict_color = verdict_data.get('color', 'yellow')
                    verdict_reasons = verdict_data.get('reasons', [])
                    
                    if verdict_color == 'green':
                        st.success(f"**Verdict: GREEN**")
                    elif verdict_color == 'yellow':
                        st.warning(f"**Verdict: YELLOW**")
                    else:
                        st.error(f"**Verdict: RED**")
                    
                    for reason in verdict_reasons:
                        st.write(f"• {reason}")
                else:
                    st.caption("Verdict not available for this run. Re-run to generate verdict data.")
                
                # === Quality Score Breakdown ===
                if composite_score_data:
                    st.subheader("Quality Score Breakdown")
                    score_col1, score_col2 = st.columns([1, 3])
                    with score_col1:
                        grade = composite_score_data.get('grade', 'N/A')
                        total_score = composite_score_data.get('total_score', 0)
                        grade_color = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'F': 'red'}.get(grade, 'gray')
                        st.markdown(f"<h1 style='color: {grade_color}; text-align: center;'>{grade}</h1>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center;'>{total_score:.0f}/100</p>", unsafe_allow_html=True)
                    
                    with score_col2:
                        breakdown = composite_score_data.get('breakdown', {})
                        if breakdown:
                            breakdown_rows = []
                            for name, data in breakdown.items():
                                breakdown_rows.append({
                                    'Metric': name.replace('_', ' ').title(),
                                    'Score': data.get('score', 0),
                                    'Weight': f"{data.get('weight', 0) * 100:.0f}%",
                                    'Weighted': data.get('weighted', 0),
                                    'Detail': data.get('detail', '')
                                })
                            breakdown_df = pd.DataFrame(breakdown_rows)
                            st.dataframe(breakdown_df, hide_index=True, use_container_width=True)
                
                # === Headline Metrics ===
                st.subheader("Headline Metrics")
                sharpe = selected_run.get('metrics.best_sharpe', None)
                ann_ret = selected_run.get('metrics.best_ann_ret', None)
                max_dd = selected_run.get('metrics.best_max_dd', None)
                turnover = selected_run.get('metrics.best_turnover', None)
                
                # Backward compatibility: extract from summary_df if metrics not in run
                if summary_df is not None and (ann_ret is None or max_dd is None or turnover is None):
                    signal_rows = summary_df[summary_df['type'] == 'signal'] if 'type' in summary_df.columns else summary_df
                    if len(signal_rows) > 0:
                        # Find best config by sharpe
                        if 'sharpe' in signal_rows.columns:
                            best_row = signal_rows.loc[signal_rows['sharpe'].idxmax()]
                            if ann_ret is None and 'ann_ret' in best_row:
                                ann_ret = best_row['ann_ret']
                            if max_dd is None and 'max_dd' in best_row:
                                max_dd = best_row['max_dd']
                            if turnover is None and 'turnover' in best_row:
                                turnover = best_row['turnover']
                            if sharpe is None and 'sharpe' in best_row:
                                sharpe = best_row['sharpe']
                
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
                with met_col2:
                    st.metric("Annual Return", f"{ann_ret:.1%}" if ann_ret else "N/A")
                with met_col3:
                    st.metric("Max Drawdown", f"{max_dd:.1%}" if max_dd else "N/A")
                with met_col4:
                    st.metric("Turnover", f"{turnover:.1%}" if turnover else "N/A")
                
                # === Robustness Analysis ===
                st.subheader("Robustness Analysis")
                robust_col1, robust_col2 = st.columns(2)
                
                with robust_col1:
                    st.markdown("**Performance by Market Cap**")
                    if cap_breakdown is not None and len(cap_breakdown) > 0:
                        st.dataframe(cap_breakdown.style.format({
                            'Sharpe': '{:.2f}',
                            'Ann Return': '{:.2%}',
                            'Max DD': '{:.2%}',
                            'Turnover': '{:.2%}',
                        }, na_rep='N/A'), hide_index=True, use_container_width=True)
                    else:
                        st.caption("Re-run this signal to generate cap breakdown data")
                
                with robust_col2:
                    st.markdown("**Performance by Year**")
                    if year_breakdown is not None and len(year_breakdown) > 0:
                        st.dataframe(year_breakdown.style.format({
                            'Sharpe': '{:.2f}',
                            'Ann Return': '{:.2%}',
                            'Max DD': '{:.2%}',
                        }, na_rep='N/A'), hide_index=True, use_container_width=True)
                    else:
                        st.caption("Re-run this signal to generate year breakdown data")
                
                st.divider()
                
                # === Experiment Details ===
                st.subheader("Experiment Details")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Signal", selected_run.get('tags.signal_name', 'N/A'))
                with col2:
                    st.metric("Data Snapshot", selected_run.get('tags.snapshot_id', 'N/A'))
                with col3:
                    st.metric("Best Sharpe", f"{sharpe:.2f}" if sharpe else "N/A")
                with col4:
                    st.metric("Git SHA", selected_run.get('tags.git_sha', 'N/A'))
                
                # Run Info expander
                with st.expander("Reproducibility Info", expanded=False):
                    st.markdown("**Run ID:**")
                    st.code(run_id, language=None)
                    st.markdown("**Signal Hash:**")
                    st.code(selected_run.get('tags.signal_hash', 'N/A'), language=None)
                
                # Suite Summary
                if summary_df is not None:
                    st.subheader("Suite Summary")
                    signal_summary = summary_df[summary_df['type'] == 'signal'] if 'type' in summary_df.columns else summary_df
                    st.dataframe(
                        signal_summary.style.format({
                            'sharpe': '{:.2f}',
                            'ann_ret': '{:.2%}',
                            'max_dd': '{:.2%}',
                            'turnover': '{:.2%}',
                        }, na_rep='N/A'),
                        use_container_width=True
                    )
                
                # Cumulative Returns
                st.subheader("Cumulative Returns")
                if daily_df is not None and 'date' in daily_df.columns and 'cumret' in daily_df.columns:
                    if 'config' in daily_df.columns:
                        if 'type' in daily_df.columns:
                            signal_configs = daily_df[daily_df['type'] == 'signal']['config'].unique().tolist()
                            baseline_configs = daily_df[daily_df['type'] == 'baseline']['config'].unique().tolist()
                        else:
                            signal_configs = daily_df['config'].unique().tolist()
                            baseline_configs = []
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_signals = st.multiselect(
                                "Signal configs", signal_configs, 
                                default=signal_configs, key="hist_signals"
                            )
                        with col2:
                            selected_baselines = st.multiselect(
                                "Baseline signals", baseline_configs,
                                default=baseline_configs, key="hist_baselines_multi"
                            )
                        
                        selected_configs = selected_signals + selected_baselines
                        filtered_df = daily_df[daily_df['config'].isin(selected_configs)]
                        
                        if 'type' in filtered_df.columns and len(filtered_df) > 0:
                            filtered_df = filtered_df.copy()
                            filtered_df['label'] = filtered_df.apply(
                                lambda r: f"📊 {r['config']}" if r['type'] == 'signal' else f"📈 {r['config']}", 
                                axis=1
                            )
                            fig = px.line(filtered_df, x='date', y='cumret', color='label',
                                         title='Cumulative Return: Signal vs Baselines',
                                         line_dash='type')
                        else:
                            fig = px.line(filtered_df, x='date', y='cumret', color='config',
                                         title='Cumulative Return by Config')
                    else:
                        fig = px.line(daily_df, x='date', y='cumret', title='Cumulative Return')
                    
                    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig, use_container_width=True, key="history_cumret")
                else:
                    st.info("No daily returns data found for this run.")
                
                # Signal Quality Charts
                st.subheader("Signal Quality Charts")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Factor Exposure bars
                    if factor_exposures is not None and len(factor_exposures) > 0:
                        factor_fig = plot_factor_exposure_bars(factor_exposures)
                        st.plotly_chart(factor_fig, use_container_width=True, key="history_factor_exposure")
                    else:
                        st.info("No factor exposure data")
                
                with chart_col2:
                    # Coverage metrics
                    if coverage:
                        st.markdown("**Coverage Metrics**")
                        cov_col1, cov_col2 = st.columns(2)
                        with cov_col1:
                            st.metric("Avg Securities/Day", f"{coverage.get('avg_securities_per_day', 0):.0f}")
                            st.metric("Unique Securities", coverage.get('unique_securities', 'N/A'))
                        with cov_col2:
                            st.metric("Total Days", coverage.get('total_days', 'N/A'))
                            if coverage.get('coverage_pct'):
                                st.metric("Universe Coverage", f"{coverage.get('coverage_pct', 0):.1f}%")
                
                # Information Coefficient (IC)
                st.subheader("Information Coefficient (IC)")
                if ic_stats:
                    ic_col1, ic_col2, ic_col3, ic_col4 = st.columns(4)
                    with ic_col1:
                        st.metric("Mean IC", f"{ic_stats.get('mean', 0):.4f}")
                    with ic_col2:
                        st.metric("t-Statistic", f"{ic_stats.get('t_stat', 0):.2f}")
                    with ic_col3:
                        st.metric("Hit Rate", f"{ic_stats.get('hit_rate', 0):.1f}%")
                    with ic_col4:
                        st.metric("Info Ratio", f"{ic_stats.get('ir', 0):.2f}")
                    
                    if ic_series is not None and len(ic_series) > 0:
                        ic_fig = plot_ic_over_time(ic_series)
                        st.plotly_chart(ic_fig, use_container_width=True, key="history_ic")
                else:
                    st.info("No IC data available for this run.")
                
                # Baseline Correlations
                st.subheader("Baseline Correlations")
                if correlations is not None and len(correlations) > 0:
                    st.dataframe(correlations.style.format({
                        'signal_corr': '{:.3f}',
                        'pnl_corr': '{:.3f}',
                    }, na_rep='N/A'), use_container_width=True)
                else:
                    st.info("No baseline correlations available.")
                
                # Lag Sensitivity Chart
                if summary_df is not None:
                    st.subheader("Lag Sensitivity")
                    lag_fig = plot_lag_sensitivity_from_summary(summary_df, show_turnover=False)
                    st.plotly_chart(lag_fig, use_container_width=True, key="history_lag_sensitivity")
                    
            except Exception as e:
                st.warning(f"Could not load artifacts: {e}")
                import traceback
                st.code(traceback.format_exc())
            
            st.markdown("[Open MLflow UI](http://localhost:5000)")


if __name__ == "__main__":
    main()
