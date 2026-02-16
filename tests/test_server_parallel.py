"""
Test multi-process backtest service with persistent workers.

Architecture:
- N worker processes, each loads master data at startup
- Workers start SEQUENTIALLY to avoid parallel memory spikes
- Workers stay alive and process signals from queue
- Signals passed as file paths (no DataFrame serialization)
- True parallelism (no GIL limitation)

Run:
    python tests/test_multiprocess.py
    python tests/test_multiprocess.py --n-workers 2 --n-signals 4
"""

import sys
import os
import time
import argparse
import gc
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
from queue import Empty

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_gb():
    """Get current process RSS in GB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except ImportError:
        # Fallback to resource module
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: bytes, Linux: KB
        if sys.platform == 'darwin':
            return usage.ru_maxrss / (1024**3)
        else:
            return usage.ru_maxrss / (1024**2)


SNAPSHOT = "2026-02-10-v1"
# Use 10 years for full testing
START_DATE = "2012-01-01"
END_DATE = "2021-12-31"  # 10 years
SIGNAL_FILE = "data/reversal_signal_analyst.csv"


def load_signal_from_csv(signal_path: str, start_date: str, end_date: str, noise_seed: int = None) -> pd.DataFrame:
    """Load signal from CSV, optionally add noise."""
    sig = pd.read_csv(signal_path)
    sig['date_sig'] = pd.to_datetime(sig['date_sig'])
    sig['date_avail'] = pd.to_datetime(sig['date_avail'])
    
    sig = sig[
        (sig['date_sig'] >= start_date) &
        (sig['date_sig'] <= end_date)
    ].copy()
    
    sig['date_ret'] = sig['date_sig'] + pd.Timedelta(days=1)
    
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        sig['signal'] = sig['signal'] + rng.normal(0, 0.01, size=len(sig))
    
    return sig


def serialize_signal_arrow(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to Arrow IPC bytes (fast, compact)."""
    import polars as pl
    import io
    
    pl_df = pl.from_pandas(df)
    buffer = io.BytesIO()
    pl_df.write_ipc(buffer)
    return buffer.getvalue()


def deserialize_signal_arrow(data: bytes) -> pd.DataFrame:
    """Deserialize Arrow IPC bytes back to DataFrame."""
    import polars as pl
    import io
    
    buffer = io.BytesIO(data)
    pl_df = pl.read_ipc(buffer)
    return pl_df.to_pandas()


def worker_process(worker_id: int, task_queue: Queue, result_queue: Queue, ready_queue: Queue, use_arrow: bool = False):
    """
    Worker process: loads data once, processes signals until shutdown.
    
    Args:
        use_arrow: If True, expect Arrow IPC bytes in task queue instead of CSV paths
    """
    from api import BacktestService
    
    mem_start = get_memory_gb()
    print(f"  [Worker {worker_id}] Starting, RSS: {mem_start:.2f}GB")
    
    # Load data (expensive, but only once per worker)
    t0 = time.perf_counter()
    service = BacktestService.get(
        SNAPSHOT,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    load_time = time.perf_counter() - t0
    
    # Force garbage collection after load
    gc.collect()
    
    mem_after_load = get_memory_gb()
    
    # Print memory breakdown
    master_gb = service._master.memory_usage(deep=True).sum() / (1024**3)
    dates_gb = service._dates.memory_usage(deep=True).sum() / (1024**3)
    print(f"  [Worker {worker_id}] Loaded in {load_time:.1f}s")
    print(f"  [Worker {worker_id}] RSS: {mem_after_load:.2f}GB, master: {master_gb:.2f}GB, dates: {dates_gb:.4f}GB")
    
    # Signal ready with memory info
    ready_queue.put((worker_id, load_time, len(service._master), mem_after_load))
    
    # Process tasks until shutdown
    while True:
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            continue
        
        if task is None:  # Shutdown signal
            break
        
        task_id, signal_data, noise_seed = task
        
        mem_before = get_memory_gb()
        
        # Get signal DataFrame
        if use_arrow:
            # Deserialize Arrow IPC bytes (fast)
            signal = deserialize_signal_arrow(signal_data)
        else:
            # Read from CSV path (slower but no main process overhead)
            signal = load_signal_from_csv(signal_data, START_DATE, END_DATE, noise_seed)
        
        # Run backtest
        t0 = time.perf_counter()
        result = service.run(signal, sigvar='signal', byvar_list=['overall'])
        elapsed = time.perf_counter() - t0
        
        # Free signal to reduce memory growth
        del signal
        gc.collect()
        
        mem_after = get_memory_gb()
        
        sharpe = result[0].iloc[0]['sharpe_ret']
        result_queue.put((task_id, worker_id, sharpe, elapsed, mem_after - mem_before))


class BacktestWorkerPool:
    """
    Pool of persistent backtest workers.
    
    Workers are started SEQUENTIALLY to avoid parallel memory spikes.
    Supports two modes:
    - CSV path mode: Pass file paths, workers read CSV locally
    - Arrow mode: Pass serialized Polars DataFrames (faster transfer)
    """
    
    def __init__(self, n_workers: int = 2, use_arrow: bool = False):
        self.n_workers = n_workers
        self.use_arrow = use_arrow
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.ready_queue = Queue()
        self.workers = []
        self.task_counter = 0
        self.pending_tasks = 0
    
    def start(self, sequential: bool = True):
        """
        Start worker processes.
        
        Args:
            sequential: If True, wait for each worker to finish loading before
                       starting the next. Prevents parallel memory spikes.
        """
        mode_str = "Arrow" if self.use_arrow else "CSV path"
        print(f"Starting {self.n_workers} worker processes ({mode_str} mode)...")
        startup_mode = "SEQUENTIAL" if sequential else "PARALLEL"
        print(f"  Startup: {startup_mode} (avoid memory spikes)")
        t0 = time.perf_counter()
        
        total_mem = 0
        for i in range(self.n_workers):
            p = Process(
                target=worker_process,
                args=(i, self.task_queue, self.result_queue, self.ready_queue, self.use_arrow)
            )
            p.start()
            self.workers.append(p)
            
            if sequential:
                # Wait for this worker to be ready before starting next
                worker_id, load_time, n_rows, mem_gb = self.ready_queue.get()
                total_mem += mem_gb
                print(f"  Worker {worker_id} ready: {load_time:.1f}s, {n_rows:,} rows, {mem_gb:.2f}GB")
        
        if not sequential:
            # Wait for all workers if parallel mode
            ready_count = 0
            while ready_count < self.n_workers:
                worker_id, load_time, n_rows, mem_gb = self.ready_queue.get()
                ready_count += 1
                total_mem += mem_gb
                print(f"  Worker {worker_id} ready: {load_time:.1f}s, {n_rows:,} rows, {mem_gb:.2f}GB")
        
        total_time = time.perf_counter() - t0
        print(f"All workers ready in {total_time:.1f}s")
        print(f"Total worker memory: {total_mem:.2f}GB")
        return total_mem
    
    def submit(self, signal_data, noise_seed: int = None) -> int:
        """
        Submit a backtest task.
        
        Args:
            signal_data: CSV path (str) or Arrow bytes (bytes) depending on mode
            noise_seed: Random seed for signal noise (only used in CSV mode)
        
        Returns:
            Task ID
        """
        task_id = self.task_counter
        self.task_counter += 1
        self.pending_tasks += 1
        self.task_queue.put((task_id, signal_data, noise_seed))
        return task_id
    
    def collect_one(self, timeout: float = None):
        """Collect one result."""
        result = self.result_queue.get(timeout=timeout)
        self.pending_tasks -= 1
        return result
    
    def collect_all(self):
        """Collect all pending results."""
        results = []
        while self.pending_tasks > 0:
            results.append(self.collect_one())
        return sorted(results, key=lambda x: x[0])  # Sort by task_id
    
    def shutdown(self):
        """Shutdown all workers."""
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join()
        print("All workers shutdown")


def main():
    parser = argparse.ArgumentParser(description="Test multi-process backtest service")
    parser.add_argument("--n-workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument("--n-signals", type=int, default=4, help="Number of signals to test")
    parser.add_argument("--parallel-start", action="store_true", help="Start workers in parallel (may spike memory)")
    parser.add_argument("--arrow", action="store_true", help="Use Arrow IPC instead of CSV paths (faster transfer)")
    args = parser.parse_args()
    
    transfer_mode = "Arrow IPC" if args.arrow else "CSV path"
    
    print("="*70)
    print(f"MULTI-PROCESS BACKTEST SERVICE")
    print(f"  Workers: {args.n_workers}")
    print(f"  Signals: {args.n_signals}")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Transfer mode: {transfer_mode}")
    print("="*70)
    
    # =========================================================================
    # 1. Start worker pool
    # =========================================================================
    print("\n1. STARTING WORKER POOL")
    print("-" * 50)
    
    pool_start = time.perf_counter()
    pool = BacktestWorkerPool(n_workers=args.n_workers, use_arrow=args.arrow)
    total_mem = pool.start(sequential=not args.parallel_start)
    startup_time = time.perf_counter() - pool_start
    
    # =========================================================================
    # 2. Prepare and submit tasks
    # =========================================================================
    print(f"\n2. SUBMITTING {args.n_signals} TASKS")
    print("-" * 50)
    
    if args.arrow:
        # Pre-load and serialize signals in main process
        print("  Preparing Arrow-serialized signals...")
        t0 = time.perf_counter()
        
        base_signal = load_signal_from_csv(SIGNAL_FILE, START_DATE, END_DATE)
        signals_arrow = []
        
        for i in range(args.n_signals):
            sig = base_signal.copy()
            if i > 0:
                rng = np.random.default_rng(42 + i)
                sig['signal'] = sig['signal'] + rng.normal(0, 0.01, size=len(sig))
            signals_arrow.append(serialize_signal_arrow(sig))
        
        prep_time = time.perf_counter() - t0
        arrow_size_mb = len(signals_arrow[0]) / (1024 * 1024)
        print(f"  Prepared {args.n_signals} signals in {prep_time:.2f}s ({arrow_size_mb:.1f}MB each)")
        
        t0 = time.perf_counter()
        for arrow_bytes in signals_arrow:
            pool.submit(arrow_bytes)
        submit_time = time.perf_counter() - t0
    else:
        # CSV path mode - just pass file paths
        noise_seeds = [None] + [42 + i for i in range(1, args.n_signals)]
        
        t0 = time.perf_counter()
        for seed in noise_seeds:
            pool.submit(SIGNAL_FILE, noise_seed=seed)
        submit_time = time.perf_counter() - t0
    
    print(f"Submitted {args.n_signals} tasks in {submit_time*1000:.1f}ms")
    
    # =========================================================================
    # 3. Collect results (workers process in parallel)
    # =========================================================================
    print(f"\n3. PROCESSING (true parallelism)")
    print("-" * 50)
    
    t0 = time.perf_counter()
    results = pool.collect_all()
    process_time = time.perf_counter() - t0
    
    for task_id, worker_id, sharpe, elapsed, mem_delta in results:
        print(f"  Task {task_id}: worker={worker_id}, sharpe={sharpe:.2f}, time={elapsed:.2f}s, mem_delta={mem_delta:+.2f}GB")
    
    # =========================================================================
    # 4. Shutdown
    # =========================================================================
    print(f"\n4. SHUTDOWN")
    print("-" * 50)
    pool.shutdown()
    
    # =========================================================================
    # Summary
    # =========================================================================
    effective_time_per_signal = process_time / len(results)
    throughput = len(results) / process_time
    
    transfer_desc = "Arrow IPC bytes (fast)" if args.arrow else "CSV path (worker reads)"
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
| Metric | Value |
|--------|-------|
| Workers | {args.n_workers} |
| Signals processed | {args.n_signals} |
| Total worker memory | {total_mem:.2f}GB |
| Startup time (one-time) | {startup_time:.1f}s |
| Processing time | {process_time:.1f}s |
| **Effective time/signal** | **{effective_time_per_signal:.2f}s** |
| Throughput | {throughput:.2f} signals/sec |

Data flow:
  - Each worker preloads master data at startup (stays in memory)
  - Signal passed via {transfer_desc}
  - Worker runs backtest with preloaded master
""")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
