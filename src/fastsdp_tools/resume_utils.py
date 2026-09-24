from pathlib import Path
import datetime
import pandas as pd


def find_run_yaml(run_folder: Path) -> Path:
    """Find the YAML config copied into a run folder (searches current dir then parent)."""
    for folder in [run_folder, run_folder.parent]:
        yamls = list(folder.glob("*.yaml"))
        if yamls:
            return yamls[0]
    raise FileNotFoundError(f"No YAML config found in {run_folder} or its parent.")


def _collect_done_from_csv(csv_path: Path, is_incomplete, fully_done: set, done_pairs: set) -> None:
    """Read one results-style CSV (data_index, target, status columns) and merge its
    completed rows into `fully_done`/`done_pairs` in place. `is_incomplete(status_str)`
    decides which rows do NOT represent a finished solve (and are therefore skipped)."""
    try:
        df = pd.read_csv(csv_path, usecols=["data_index", "target", "status"])
        df = df[~df["status"].astype(str).apply(is_incomplete)]
        for _, row in df.iterrows():
            if pd.isna(row["data_index"]):
                continue
            idx = int(row["data_index"])
            if pd.isna(row["target"]):
                fully_done.add(idx)
            else:
                done_pairs.add((idx, int(row["target"])))
    except (KeyError, pd.errors.EmptyDataError, ValueError):
        pass


def find_processed_indices(run_folder: Path) -> tuple:
    """Return (fully_done, done_pairs).

    fully_done: set of data_index where target is NaN — UntargetedSDP rows (or a
                Targeted "trivially_solved" probe) where the entire sample is solved
                in one shot, so no partial state is possible.
    done_pairs: set of (data_index, target) int tuples — TargetedSDP rows where each
                target class is a separate SDP solve and partial completion can occur.

    Scans both the classic `results.csv` (rows with status "pre-solve" or "crashed"
    excluded : written before/during optimization, not a completed solve) and the
    experimental conic-bundle path's `results_conic_bundle.csv` (certification_problem.py
    run_conic_bundle — same data_index/target/status schema, but no "pre-solve"/"crashed"
    placeholder rows since it only writes after a target fully returns; rows with a
    status starting with "error" are excluded instead, so a run that raised an exception
    gets retried on --resume rather than being treated as done).
    """
    _INCOMPLETE_STATUSES = {"pre-solve", "crashed"}
    fully_done = set()
    done_pairs = set()
    for csv_path in run_folder.rglob("results.csv"):
        _collect_done_from_csv(
            csv_path, lambda s: s in _INCOMPLETE_STATUSES, fully_done, done_pairs
        )
    for csv_path in run_folder.rglob("results_conic_bundle.csv"):
        _collect_done_from_csv(
            csv_path, lambda s: s.startswith("error"), fully_done, done_pairs
        )
    return fully_done, done_pairs


def format_intervals(indices: set) -> str:
    """Format a set of integers as a union of closed intervals, e.g. [0,51]∪[200,247]."""
    if not indices:
        return "∅"
    sorted_idx = sorted(indices)
    intervals = []
    start = end = sorted_idx[0]
    for i in sorted_idx[1:]:
        if i == end + 1:
            end = i
        else:
            intervals.append((start, end))
            start = end = i
    intervals.append((start, end))
    return "∪".join(f"[{a},{b}]" for a, b in intervals)


def log_run_history(run_folder: Path, label: str, start_time: datetime.datetime, indices: set) -> None:
    """Append one line to run_history.txt: run number, label, timestamp, and processed intervals."""
    history_path = run_folder / "run_history.txt"
    run_number = 1
    if history_path.exists():
        with open(history_path) as f:
            run_number = sum(1 for line in f if line.strip()) + 1
    ts_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    interval_str = format_intervals(indices)
    with open(history_path, "a") as f:
        f.write(f"Run {run_number} ({label}): {ts_str}, indices {interval_str}\n")


def load_existing_results(run_folder: Path) -> pd.DataFrame:
    """Concatenate all results.csv files found recursively under run_folder."""
    dfs = []
    for csv_path in run_folder.rglob("results.csv"):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                dfs.append(df)
        except pd.errors.EmptyDataError:
            pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
