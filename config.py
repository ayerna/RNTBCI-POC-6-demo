# ─── BMS Pipeline Configuration ───────────────────────────────────────────────
#
# All paths are resolved relative to THIS file so the project works on any
# machine after cloning — no edits needed unless you place the CSV elsewhere.
#
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Dataset ────────────────────────────────────────────────────────────────────
# Default: looks for the CSV in the same folder as this script.
# Override with the CSV_PATH environment variable if stored elsewhere:
#   set CSV_PATH=C:\data\bms_processed_1hz.csv  (Windows)
#   export CSV_PATH=/data/bms_processed_1hz.csv  (Linux/Mac)
#
# If the file does not exist, run:  python generate_sample_data.py
CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(_HERE, "bms_processed_1hz.csv")
)

# ── Trip Simulation ────────────────────────────────────────────────────────────
TRIP_HOURS      = 2                        # simulated trip duration in hours
TRIP_ROWS       = TRIP_HOURS * 3600        # 1 Hz -> rows == seconds
TRIP_START_MODE = "discharge"              # "start" | "discharge" | "random"
#   "start"     -> use first TRIP_ROWS rows
#   "discharge" -> find first heavy-discharge segment (most interesting)
#   "random"    -> pick a random valid start

STREAM_DELAY    = 0.0           # seconds between rows (0=fast-sim, 1.0=real-time)
WINDOW_SIZE     = 30            # rows per analysis window (30 s at 1 Hz)
BUFFER_MAX      = 3600          # rolling buffer max size (1 hour of readings)

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "llama3:latest"
LLM_TIMEOUT_S   = 45           # seconds before fallback rule engine kicks in
LLM_MAX_TOKENS  = 60           # keep answers to ~2 sentences (fast + concise)
LLM_TEMPERATURE = 0.2          # low = more deterministic, less hallucination

# ── Session ───────────────────────────────────────────────────────────────────
DRIVER_NAME     = "Driver"     # overridden at startup by user input

# ── Storage ───────────────────────────────────────────────────────────────────
KB_JSON_PATH    = os.path.join(_HERE, "knowledge_base.json")

# ── Speech ────────────────────────────────────────────────────────────────────
WHISPER_MODEL   = "base"        # tiny | base | small | medium
MIC_RECORD_SECS = 6             # seconds of audio captured per voice query
TTS_RATE        = 165           # pyttsx3 speech rate (words per minute)
TTS_VOLUME      = 1.0           # 0.0 - 1.0
SPEECH_ENABLED  = True          # set False to disable TTS/STT entirely

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "high_temp":          35.0,   # deg C
    "critical_temp":      37.0,   # deg C
    "high_current":      150.0,   # A  (discharge)
    "regen_current":     -50.0,   # A  (negative = regen / charging)
    "low_soc":            25.0,   # %
    "critical_soc":       18.0,   # %
    "rapid_soc_drop":     -0.5,   # % per second (window average rate)
    "high_power":      80_000.0,  # W
    "low_voltage":       370.0,   # V
    "high_voltage":      445.0,   # V
    "current_spike":     400.0,   # A  single-sample spike threshold
}
