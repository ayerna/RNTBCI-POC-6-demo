"""
data_injector.py - Layer 1 / 2
Loads CSV once, slices a realistic trip, streams rows into a shared deque buffer.
"""

import time
import threading
import random
import pandas as pd
from collections import deque
from datetime import datetime

from config import (CSV_PATH, TRIP_ROWS, TRIP_START_MODE,
                    STREAM_DELAY, BUFFER_MAX)
import logger

# ── Shared state ───────────────────────────────────────────────────────────────
_buffer: deque       = deque(maxlen=BUFFER_MAX)
_buffer_lock         = threading.Lock()
_inject_done         = threading.Event()
_row_count           = 0
_trip_meta: dict     = {}      # store trip start/end timestamps for display


# ── Public API ─────────────────────────────────────────────────────────────────

def get_buffer_snapshot() -> list[dict]:
    with _buffer_lock:
        return list(_buffer)


def get_latest(n: int = 30) -> list[dict]:
    with _buffer_lock:
        data = list(_buffer)
    return data[-n:] if len(data) >= n else data


def get_row_count() -> int:
    return _row_count


def injection_complete() -> bool:
    return _inject_done.is_set()


def get_trip_meta() -> dict:
    return dict(_trip_meta)


# ── Trip Selection ─────────────────────────────────────────────────────────────

def load_trip_data() -> pd.DataFrame:
    """
    Load CSV exactly once and slice a TRIP_ROWS-length segment.

    TRIP_START_MODE controls selection strategy:
      'discharge' -> finds first sustained high-discharge segment (most realistic)
      'random'    -> picks a random valid start index
      'start'     -> uses the first TRIP_ROWS rows
    """
    logger.log_system(f"Loading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
    df = df.rename(columns={
        "datetime": "timestamp",
        "Curr":     "current",
        "Volt":     "voltage",
        "SoC":      "soc",
        "Temp":     "temperature",
        "Power_W":  "power_w",
        "State":    "state",
    })

    total = len(df)
    mode  = TRIP_START_MODE.lower()

    if mode == "discharge" and total > TRIP_ROWS:
        # Find first index where current > 20 A for at least TRIP_ROWS consecutive rows
        discharge_mask = df["current"] > 20
        start_idx = _find_run_start(discharge_mask, TRIP_ROWS)
        if start_idx is None:
            # Fallback: just use biggest-current region
            start_idx = (df["current"].rolling(TRIP_ROWS, min_periods=1)
                           .mean().idxmax())
            start_idx = max(0, start_idx - TRIP_ROWS // 2)
    elif mode == "random" and total > TRIP_ROWS:
        start_idx = random.randint(0, total - TRIP_ROWS - 1)
    else:
        start_idx = 0

    trip_df = df.iloc[start_idx : start_idx + TRIP_ROWS].copy().reset_index(drop=True)

    ts_start = str(trip_df["timestamp"].iloc[0])
    ts_end   = str(trip_df["timestamp"].iloc[-1])
    _trip_meta.update({
        "start": ts_start, "end": ts_end,
        "n_rows": len(trip_df), "mode": mode,
        "start_idx_in_csv": start_idx,
    })

    logger.show_trip_banner(ts_start, ts_end, len(trip_df), mode)
    return trip_df


def _find_run_start(mask: pd.Series, min_run: int) -> int | None:
    """Return first index where True runs for at least min_run consecutive rows."""
    count = 0
    for i, val in enumerate(mask):
        if val:
            count += 1
            if count >= min_run:
                return i - min_run + 1
        else:
            count = 0
    return None


# ── Streaming ──────────────────────────────────────────────────────────────────

def _stream_rows(trip_df: pd.DataFrame, on_row_callback=None) -> None:
    global _row_count

    cols = ["timestamp", "voltage", "current", "soc", "temperature",
            "power_w", "state"]

    for _, row in trip_df.iterrows():
        record = {c: row[c] for c in cols if c in row.index}
        record["injected_at"] = datetime.now().isoformat()

        with _buffer_lock:
            _buffer.append(record)
        _row_count += 1

        if on_row_callback:
            on_row_callback(record, _row_count)

        if STREAM_DELAY > 0:
            time.sleep(STREAM_DELAY)

    _inject_done.set()
    logger.log_system("All trip rows injected into buffer.")


def start_injection(trip_df: pd.DataFrame,
                    on_row_callback=None) -> threading.Thread:
    """Start streaming in a daemon background thread."""
    t = threading.Thread(
        target=_stream_rows,
        args=(trip_df, on_row_callback),
        daemon=True,
        name="BMSInjector",
    )
    t.start()
    logger.log_system(f"Injection thread started ({len(trip_df):,} rows queued).")
    return t
