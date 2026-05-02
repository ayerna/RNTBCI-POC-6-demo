"""
windowing.py - Layer 3 / 4
Converts fixed-size time windows into rich, human-readable KB entries.
Observations are written in natural language; raw numbers only appear in metrics.
"""

import math
from datetime import datetime
from config import WINDOW_SIZE, THRESHOLDS
import logger


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_vals(values: list, strip_nan: bool = True) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
            if strip_nan and math.isnan(f):
                continue
            out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def _avg(v):  return sum(v) / len(v) if v else 0.0
def _mx(v):   return max(v) if v else 0.0
def _mn(v):   return min(v) if v else 0.0


def _trend(values: list[float]) -> str:
    """
    Linear regression slope -> 'rising' | 'falling' | 'stable'.
    'stable' if relative change < 0.5 % of the mean.
    """
    n = len(values)
    if n < 4:
        return "stable"
    xs  = list(range(n))
    mx  = n / 2
    my  = _avg(values)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0 or my == 0:
        return "stable"
    rel = (num / den) / abs(my)        # normalised slope
    if abs(rel) < 0.005:
        return "stable"
    return "rising" if rel > 0 else "falling"


def _rate_of_change(values: list[float]) -> float:
    """Total change divided by number of steps."""
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / len(values)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS WINDOW  (Layer 3)
# ─────────────────────────────────────────────────────────────────────────────

def process_window(window: list[dict]) -> dict | None:
    if not window:
        return None

    voltages     = _safe_vals([r["voltage"]      for r in window])
    currents     = _safe_vals([r["current"]      for r in window])
    socs         = _safe_vals([r["soc"]          for r in window])
    temperatures = _safe_vals([r["temperature"]  for r in window])
    powers       = _safe_vals([r.get("power_w", 0) for r in window])

    soc_rate = _rate_of_change(socs)           # %/s over window

    states = [r.get("state", "Unknown") for r in window]
    state_counts: dict[str, int] = {}
    for s in states:
        state_counts[s] = state_counts.get(s, 0) + 1
    dominant_state = max(state_counts, key=state_counts.get)

    return {
        "t_start":  str(window[0].get("timestamp", "")),
        "t_end":    str(window[-1].get("timestamp", "")),
        "n_rows":   len(window),
        "stats": {
            "voltage":     {"avg": _avg(voltages),     "min": _mn(voltages),     "max": _mx(voltages)},
            "current":     {"avg": _avg(currents),     "min": _mn(currents),     "max": _mx(currents)},
            "soc":         {"avg": _avg(socs),         "min": _mn(socs),         "max": _mx(socs)},
            "temperature": {"avg": _avg(temperatures), "min": _mn(temperatures), "max": _mx(temperatures)},
            "power_w":     {"avg": _avg(powers),       "max": _mx(powers)},
        },
        "trends": {
            "voltage":          _trend(voltages),
            "current":          _trend(currents),
            "soc":              _trend(socs),
            "temperature":      _trend(temperatures),
            "soc_rate_pct_s":   round(soc_rate, 5),
        },
        "dominant_state": dominant_state,
        "state_counts":   state_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE KB ENTRY  (Layer 4) — human-readable observations
# ─────────────────────────────────────────────────────────────────────────────

def generate_kb_entry(window_id: int, summary: dict) -> dict:
    T  = THRESHOLDS
    s  = summary["stats"]
    tr = summary["trends"]

    avg_v   = s["voltage"]["avg"];     min_v = s["voltage"]["min"]
    avg_c   = s["current"]["avg"];     max_c = s["current"]["max"];  min_c = s["current"]["min"]
    avg_soc = s["soc"]["avg"];         min_soc = s["soc"]["min"]
    avg_t   = s["temperature"]["avg"]; max_t   = s["temperature"]["max"]
    avg_p   = s["power_w"]["avg"]
    soc_rate = tr["soc_rate_pct_s"]
    dom      = summary["dominant_state"]

    observations: list[str] = []
    flags:        list[str] = []

    # ── Voltage ───────────────────────────────────────────────────────────────
    if avg_v < T["low_voltage"]:
        observations.append("Pack voltage has dropped to critically low levels — possible deep discharge")
        flags.append("LOW_VOLTAGE")
    elif avg_v > T["high_voltage"]:
        observations.append("Pack voltage is elevated above normal — likely peak from regenerative braking")
        flags.append("HIGH_VOLTAGE")
    else:
        observations.append(f"Pack voltage is holding steady in the normal operating band")

    if tr["voltage"] == "falling":
        observations.append("Voltage trending downward across this window — battery under sustained load")
    elif tr["voltage"] == "rising":
        observations.append("Voltage recovering during this window — load reduced or regen active")

    # ── Current ───────────────────────────────────────────────────────────────
    if avg_c > T["high_current"]:
        observations.append("Battery is experiencing high average discharge current — heavy acceleration or sustained high speed")
        flags.append("HIGH_CURRENT")
    elif avg_c < 5:
        observations.append("Very low average current draw — vehicle likely idle or coasting")
    else:
        observations.append("Current draw is moderate — typical cruising or light acceleration")

    if max_c > T["current_spike"]:
        observations.append(f"Sudden current spike detected — sharp acceleration burst observed")
        flags.append("CURRENT_SPIKE")

    if min_c < T["regen_current"]:
        observations.append("Strong regenerative braking events captured — energy being recovered efficiently")
        flags.append("REGEN_BRAKING")

    if tr["current"] == "rising":
        observations.append("Current demand increasing over this window — driving becoming more aggressive")
    elif tr["current"] == "falling":
        observations.append("Current demand easing off — driving becoming gentler")

    # ── State of Charge ───────────────────────────────────────────────────────
    if min_soc < T["critical_soc"]:
        observations.append("CRITICAL: State of Charge has fallen below the safe minimum — charge immediately")
        flags.append("CRITICAL_SOC")
    elif avg_soc < T["low_soc"]:
        observations.append("Battery is running low — range is limited, plan for charging soon")
        flags.append("LOW_SOC")
    elif avg_soc > 80:
        observations.append("Battery is well charged — substantial range available")
    else:
        observations.append(f"State of Charge is in a comfortable operating zone")

    if soc_rate < T["rapid_soc_drop"]:
        observations.append("SoC is depleting faster than expected — possible high load or aggressive driving")
        flags.append("RAPID_SOC_DROP")
    elif tr["soc"] == "rising":
        observations.append("SoC is increasing — vehicle is charging or recovering energy via regen")
    elif tr["soc"] == "falling":
        observations.append("SoC is gradually declining — normal energy consumption during driving")

    # ── Temperature ───────────────────────────────────────────────────────────
    if max_t >= T["critical_temp"]:
        observations.append(f"CRITICAL: Cell temperature has reached a dangerous peak — thermal runaway risk is elevated")
        flags.append("CRITICAL_TEMP")
    elif avg_t >= T["high_temp"]:
        observations.append("Battery pack is running hot — sustained thermal stress detected")
        flags.append("HIGH_TEMP")
    else:
        observations.append("Thermal conditions are within the safe operating range")

    if tr["temperature"] == "rising":
        observations.append("Temperature is trending upward — monitor carefully if load continues")
    elif tr["temperature"] == "falling":
        observations.append("Temperature is cooling down — thermal load has reduced")

    # ── Power ────────────────────────────────────────────────────────────────
    if avg_p > T["high_power"]:
        observations.append(f"Average power draw exceeds {T['high_power']/1000:.0f} kW — sustained high-performance driving")
        flags.append("HIGH_POWER")

    # ── Drive State ──────────────────────────────────────────────────────────
    state_desc = {
        "Discharge": "predominantly discharging — active driving",
        "Charge":    "in charging state — not driving",
        "Idle":      "idle — vehicle stationary or parked",
        "Regen":     "regenerating energy — braking or downhill",
    }
    observations.append(
        f"Drive mode is {state_desc.get(dom, dom.lower())} during this window"
    )

    # ── Inference & Severity ─────────────────────────────────────────────────
    inference, severity = _infer(flags, avg_soc, avg_t, avg_c, soc_rate, tr)

    entry = {
        "window_id":    window_id,
        "time_range":   f"{summary['t_start']} -> {summary['t_end']}",
        "n_rows":       summary["n_rows"],
        "observations": observations,
        "flags":        flags,
        "inference":    inference,
        "severity":     severity,
        "metrics": {
            "avg_voltage_V":   round(avg_v,   2),
            "min_voltage_V":   round(min_v,   2),
            "avg_current_A":   round(avg_c,   2),
            "max_current_A":   round(max_c,   2),
            "avg_soc_pct":     round(avg_soc, 2),
            "min_soc_pct":     round(min_soc, 2),
            "avg_temp_C":      round(avg_t,   2),
            "max_temp_C":      round(max_t,   2),
            "avg_power_kW":    round(avg_p / 1000, 2),
            "soc_rate_pct_s":  round(soc_rate, 5),
            "dominant_state":  dom,
        },
        "created_at": datetime.now().isoformat(),
    }

    # Rich console display
    logger.log_window(window_id, summary["n_rows"],
                      f"{summary['t_start'][-8:]} -> {summary['t_end'][-8:]}")
    logger.log_kb(window_id, severity, inference)

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED EXPERT INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _infer(flags: list[str], avg_soc: float, avg_t: float,
           avg_c: float, soc_rate: float, tr: dict) -> tuple[str, str]:

    flag_set = set(flags)
    critical = {"CRITICAL_TEMP", "CRITICAL_SOC"}
    warning  = {"HIGH_CURRENT", "HIGH_TEMP", "RAPID_SOC_DROP",
                 "LOW_SOC", "LOW_VOLTAGE", "CURRENT_SPIKE", "HIGH_POWER"}

    # Critical path
    if flag_set & critical:
        if "CRITICAL_TEMP" in flag_set and "CRITICAL_SOC" in flag_set:
            return ("Battery is in a dual-critical state: the pack is overheating while "
                    "almost empty. Stop driving immediately and allow cooling before charging.", "CRITICAL")
        if "CRITICAL_TEMP" in flag_set:
            return ("Thermal event is developing inside the pack. Reduce speed, turn off "
                    "ancillary loads, and seek service if temperature doesn't drop.", "CRITICAL")
        return ("The battery charge level is critically low. Find the nearest charging "
                "station immediately to avoid being stranded.", "CRITICAL")

    # Multiple simultaneous warnings
    active_warnings = flag_set & warning
    if len(active_warnings) >= 3:
        return ("Multiple battery stressors are occurring simultaneously: high current, "
                "elevated temperature and rapid SoC drain suggest aggressive or demanding "
                "driving conditions. Ease off for better range and battery longevity.", "Warning")

    if "HIGH_CURRENT" in flag_set and "RAPID_SOC_DROP" in flag_set:
        return ("High-load driving detected. The battery is being discharged aggressively, "
                "depleting range faster than normal. Consider a gentler driving style.", "Warning")

    if "HIGH_TEMP" in flag_set and "HIGH_CURRENT" in flag_set:
        return ("High current is generating significant heat in the pack. "
                "Sustained performance driving — monitor thermal trends.", "Warning")

    if "HIGH_TEMP" in flag_set:
        return ("Thermal stress is building up in the battery. "
                "A short cool-down period would benefit long-term cell health.", "Warning")

    if "LOW_SOC" in flag_set:
        return ("The battery is entering a low-charge zone. "
                "Plan a charging stop soon to avoid range anxiety.", "Warning")

    if "RAPID_SOC_DROP" in flag_set:
        return ("State of Charge is draining more rapidly than expected. "
                "Check for high ancillary loads or aggressive acceleration.", "Warning")

    if "REGEN_BRAKING" in flag_set and avg_c < 20:
        return ("Efficient driving with strong regenerative braking. "
                "Energy is being recovered well — good driving behaviour.", "Info")

    if avg_c < 10 and avg_soc > 40 and avg_t < 30:
        return ("Battery is operating efficiently. "
                "Current load is light and all parameters are well within limits.", "Normal")

    return ("Battery is operating within expected parameters for the current "
            "driving conditions. No immediate action required.", "Normal")
