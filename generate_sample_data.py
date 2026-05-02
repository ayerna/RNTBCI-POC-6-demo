"""
generate_sample_data.py
=======================
Generates a realistic synthetic BMS dataset (bms_processed_1hz.csv)
so anyone who clones this repo can run the pipeline without the real dataset.

Usage:
    python generate_sample_data.py

Output:
    bms_processed_1hz.csv   (same directory as this script, ~5 MB, 30,600 rows)

The generated data simulates multiple charge/discharge cycles with:
- Realistic voltage curves (higher SoC = higher voltage)
- Current spikes for acceleration, negative current for regen braking
- Temperature rising under load, cooling at rest
- SoC depletion during discharge, recovery during charge
- Drive states: Idle / Discharge / Charge / Regen
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

SEED          = 42
TOTAL_ROWS    = 30_600          # ~8.5 hours at 1 Hz
CYCLE_LEN     = 3_600           # each cycle ~1 hour
OUTPUT_FILE   = os.path.join(os.path.dirname(__file__), "bms_processed_1hz.csv")

rng = np.random.default_rng(SEED)

def _voltage_from_soc(soc: np.ndarray) -> np.ndarray:
    """Approximate OCV curve: 360 V at 0% SoC, 450 V at 100%."""
    return 360 + (soc / 100) * 90

def generate() -> pd.DataFrame:
    print(f"Generating {TOTAL_ROWS:,} rows of synthetic BMS data...")
    start_ts = datetime(2020, 1, 17, 23, 0, 0, tzinfo=timezone.utc)

    timestamps = [start_ts + timedelta(seconds=i) for i in range(TOTAL_ROWS)]

    soc        = np.zeros(TOTAL_ROWS)
    current    = np.zeros(TOTAL_ROWS)
    voltage    = np.zeros(TOTAL_ROWS)
    temperature= np.zeros(TOTAL_ROWS)
    state_arr  = ["Idle"] * TOTAL_ROWS
    cycle_arr  = np.zeros(TOTAL_ROWS, dtype=int)

    soc_val    = 80.0          # start at 80%
    temp_val   = 18.0
    cycle_id   = 1

    for i in range(TOTAL_ROWS):
        pos_in_cycle = i % CYCLE_LEN
        phase        = pos_in_cycle / CYCLE_LEN   # 0.0 -> 1.0

        cycle_id = (i // CYCLE_LEN) + 1

        # ── Drive pattern within each cycle ───────────────────────────────────
        if phase < 0.05:
            # Idle at start
            st   = "Idle"
            curr = rng.uniform(0, 3)
            soc_val = min(soc_val + 0.001, 100.0)

        elif phase < 0.55:
            # Main discharge (driving)
            if rng.random() < 0.08:
                # Regen braking event
                st   = "Regen"
                curr = -rng.uniform(30, 150)
                soc_val = min(soc_val + abs(curr) * 0.0002, 100.0)
            elif rng.random() < 0.05:
                # Acceleration spike
                st   = "Discharge"
                curr = rng.uniform(200, 450)
                soc_val -= curr * 0.00004
            else:
                st   = "Discharge"
                curr = rng.uniform(10, 160)
                soc_val -= curr * 0.00003

        elif phase < 0.60:
            # Short idle
            st   = "Idle"
            curr = rng.uniform(0, 5)

        else:
            # Charging phase
            st   = "Charge"
            curr = -rng.uniform(20, 80)
            soc_val = min(soc_val + 0.025, 96.0)

        soc_val = float(np.clip(soc_val, 5.0, 98.0))

        # ── Temperature model ─────────────────────────────────────────────────
        if st == "Discharge" and curr > 100:
            temp_val += rng.uniform(0.01, 0.08)
        elif st == "Charge":
            temp_val += rng.uniform(0.005, 0.03)
        else:
            temp_val -= rng.uniform(0.00, 0.03)
        temp_val = float(np.clip(temp_val, 12.0, 38.0))

        # ── Voltage: OCV + load sag ───────────────────────────────────────────
        ocv     = _voltage_from_soc(np.array([soc_val]))[0]
        sag     = (curr / 500) * 25   # up to -25 V sag at 500 A
        volt    = float(np.clip(ocv - sag + rng.normal(0, 0.3), 350, 460))

        soc[i]         = round(soc_val, 2)
        current[i]     = round(float(curr), 4)
        voltage[i]     = round(volt, 6)
        temperature[i] = round(temp_val, 2)
        state_arr[i]   = st
        cycle_arr[i]   = cycle_id

    # ── Derived columns ───────────────────────────────────────────────────────
    power_w   = voltage * current
    energy_wh = np.cumsum(power_w) / 3600

    dsoc_dt   = np.concatenate([[np.nan], np.diff(soc)])
    dvolt_dt  = np.concatenate([[np.nan], np.diff(voltage)])

    roll = pd.Series(current).rolling(window=30, min_periods=1)
    curr_roll_mean = roll.mean().values
    curr_roll_std  = roll.std().values

    volt_roll_mean = pd.Series(voltage).rolling(window=30, min_periods=1).mean().values
    temp_roll_mean = pd.Series(temperature).rolling(window=30, min_periods=1).mean().values

    df = pd.DataFrame({
        "datetime":       [ts.isoformat() for ts in timestamps],
        "Curr":           current,
        "Volt":           voltage,
        "SoC":            soc,
        "Temp":           temperature,
        "Power_W":        power_w,
        "Energy_Wh":      energy_wh,
        "dSoC_dt":        dsoc_dt,
        "dVolt_dt":       dvolt_dt,
        "Curr_roll_mean": curr_roll_mean,
        "Curr_roll_std":  curr_roll_std,
        "Volt_roll_mean": volt_roll_mean,
        "Temp_roll_mean": temp_roll_mean,
        "State":          state_arr,
        "cycle_id":       cycle_arr,
    })

    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / 1_048_576
    print(f"Done. Saved {len(df):,} rows to: {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print(f"SoC range  : {df['SoC'].min():.1f}% — {df['SoC'].max():.1f}%")
    print(f"Temp range : {df['Temp'].min():.1f}C — {df['Temp'].max():.1f}C")
    print(f"Curr range : {df['Curr'].min():.1f}A — {df['Curr'].max():.1f}A")
    print(f"States     : {df['State'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    generate()
