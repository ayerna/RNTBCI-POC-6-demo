# 🔋 RNTBCI-POC-5 — Real-Time EV Battery Management System (BMS) AI Pipeline

> **A fully demo-ready, interactive AI assistant for EV battery telemetry.**
> Simulates a real vehicle trip, builds a temporal knowledge base window-by-window,
> and lets drivers query battery health via voice or text — answered by a local LLM
> or an expert rule engine fallback.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Module Breakdown](#module-breakdown)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Injection Modes](#injection-modes)
- [Query Interface](#query-interface)
- [Speech Integration](#speech-integration)
- [Knowledge Base Format](#knowledge-base-format)
- [LLM Integration & Fallback](#llm-integration--fallback)
- [Configuration Reference](#configuration-reference)
- [Sample Output](#sample-output)
- [File Structure](#file-structure)

---

## Overview

This project implements a **6-layer real-time BMS telemetry pipeline** that:

1. Loads a pre-processed 1 Hz EV sensor CSV **once** and streams it row-by-row into a memory buffer
2. Groups rows into **fixed-size time windows** (30 s default)
3. Converts each window into a **structured, human-readable knowledge base entry** using a rule-based expert system
4. Stores the knowledge base in-memory with JSON persistence
5. Accepts natural-language driver queries and answers them using **Ollama (llama3)** with a rule-engine fallback
6. Supports **voice input** (Whisper STT) and **spoken responses** (pyttsx3 TTS)

The system is designed to feel like a **real vehicle AI assistant** interacting with live telemetry.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Raw Sensor Input (simulated from CSV at 1 Hz)        │
│  Columns: voltage · current · temperature · SoC · power · state │
└───────────────────────────┬─────────────────────────────────────┘
                            │ row-by-row injection (deque buffer)
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 2 — Ingestion & Buffer                                    │
│  · deque(maxlen=3600) rolling buffer                             │
│  · Three trip-selection modes: discharge / random / start        │
│  · Rich [INJECT] logs every 10th row                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ every 30 rows
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 3 — Windowing Engine                                      │
│  · Stats: avg/min/max per signal                                 │
│  · Trend: rising / falling / stable (linear regression slope)   │
│  · SoC rate-of-change per second                                 │
│  · Dominant drive state                                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 4 — Knowledge Base Generator (Expert Rule Engine)         │
│  · 12 flag types (HIGH_TEMP, CRITICAL_SOC, REGEN_BRAKING, …)    │
│  · Natural-language observations (no raw numbers in text)        │
│  · Severity: Normal / Info / Warning / CRITICAL                  │
│  · 1–2 sentence inference per window                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 5 — Knowledge Base Storage                                │
│  · Thread-safe in-memory list                                    │
│  · JSON persistence (knowledge_base.json)                        │
│  · Query helpers: get_recent(), get_by_severity(), get_flagged() │
│  · Rule-based summary generator (LLM fallback)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 6 — LLM Agent (Ollama llama3)                             │
│  · Session-aware: greets by name once, personalises all replies  │
│  · Fast-path: simple queries answered without LLM call           │
│  · Thread-based timeout (45 s) → automatic rule-engine fallback  │
│  · Strict 2-sentence format enforced in system prompt            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Driver Interface                                                │
│  · CLI: Type [T] | Speak [S] | Demo [D] | Summary [K] | Quit[Q] │
│  · Voice input:  Whisper (openai-whisper, local)                 │
│  · Voice output: pyttsx3 (offline TTS)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

| File | Layer | Responsibility |
|---|---|---|
| `config.py` | — | All constants, thresholds, paths, speech config |
| `data_injector.py` | 1 / 2 | CSV load (once), trip slicing, deque buffer, streaming thread |
| `windowing.py` | 3 / 4 | `process_window()` stats + trend engine; `generate_kb_entry()` natural-language KB builder |
| `kb_store.py` | 5 | Thread-safe KB list, JSON persistence, query helpers, rule-based fallback summary |
| `llm_agent.py` | 6 | Session object, greeting logic, context builder, Ollama query, timeout, fast-path |
| `speech.py` | + | `init_speech()`, `speak()` (pyttsx3 TTS), `listen()` (Whisper STT) |
| `logger.py` | + | Centralised Rich console: `[INJECT]`, `[WINDOW]`, `[KB]`, `[LLM]`, `[SPEECH]`, `[SYSTEM]` |
| `main.py` | — | Orchestrator: mode selection, threading, CLI loop, session init |
| `smoke_test.py` | — | Fast non-interactive validation (no LLM/speech required) |

---

## Features

### ✅ Real-Time Simulation
- CSV loaded **exactly once**; rows streamed row-by-row into a `deque(maxlen=3600)`
- No repeated full-dataset reloads
- Three trip-selection strategies ensure the most interesting (discharge-heavy) segment is used

### ✅ Expert Rule Engine
- 12 flag types detected per window:
  `HIGH_TEMP`, `CRITICAL_TEMP`, `HIGH_CURRENT`, `CURRENT_SPIKE`, `REGEN_BRAKING`,
  `LOW_SOC`, `CRITICAL_SOC`, `RAPID_SOC_DROP`, `HIGH_VOLTAGE`, `LOW_VOLTAGE`, `HIGH_POWER`
- Severity determined by flag combination logic (not single thresholds)
- Observations written in plain English — no raw numbers in text

### ✅ LLM Integration
- Local Ollama (`llama3:latest`) — no API key, no internet required
- Session-aware: greets driver by name on first response, personalises all subsequent replies
- Strict 2-sentence response format enforced in system prompt
- `num_predict=60`, `temperature=0.2` — fast, deterministic, concise

### ✅ LLM Fallback
- LLM runs in a daemon thread with a 45-second timeout
- On timeout/error → automatic **rule-engine fallback** generates a plain-English answer
- Fast-path: simple queries ("is the soc okay?", "am I safe?") bypass the LLM entirely

### ✅ Voice I/O
- **STT**: Whisper `base` model — captures 6 s of microphone audio, transcribes locally
- **TTS**: pyttsx3 — speaks every LLM/fallback response aloud, offline
- Both degrade gracefully if hardware/libraries are unavailable

### ✅ Three Injection Modes
- **Mode 1**: Full auto-streaming (7 200 rows, 2-hour trip)
- **Mode 2**: Step-by-step — press Enter to inject one row at a time (proves real-time behaviour)
- **Mode 3**: Scenario injection — choose from 5 presets (Thermal Spike, Critical SOC, Aggressive Acceleration, Regen Braking, Custom)

### ✅ Rich Console Logging
- Colour-coded per layer using the `rich` library
- `[INJECT]` cyan · `[WINDOW]` bright-blue · `[KB]` green/yellow/red · `[LLM]` magenta · `[SPEECH]` bright-magenta
- Panels for trip banner, KB entries, LLM answers, trip summary table

---

## Dataset

**File:** `bms_processed_1hz.csv` (place in `d:\renaults_z\` or update `CSV_PATH` in `config.py`)

**Shape:** 30 600 rows × 15 columns at 1 Hz

| Column | Unit | Description |
|---|---|---|
| `datetime` | — | Timestamp (UTC) |
| `Curr` | A | Pack current (positive = discharge, negative = regen) |
| `Volt` | V | Pack voltage |
| `SoC` | % | State of Charge |
| `Temp` | °C | Pack temperature |
| `Power_W` | W | Instantaneous power |
| `Energy_Wh` | Wh | Cumulative energy |
| `dSoC_dt` | %/s | SoC rate of change |
| `dVolt_dt` | V/s | Voltage rate of change |
| `Curr_roll_mean` | A | Rolling mean current |
| `Curr_roll_std` | A | Rolling std current |
| `Volt_roll_mean` | V | Rolling mean voltage |
| `Temp_roll_mean` | °C | Rolling mean temperature |
| `State` | — | Drive state label (Idle/Discharge/Charge/Regen) |
| `cycle_id` | — | Charge/discharge cycle index |

> The real dataset **`bms_processed_1hz.csv`** is included in this repository (5 MB).
> A synthetic alternative can also be generated locally:
> ```bash
> python generate_sample_data.py
> ```

---

## Installation

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com/) installed and running locally
- `llama3` model pulled: `ollama pull llama3`
- A microphone (optional, for voice input)

### Install Python dependencies

```bash
pip install pandas numpy rich ollama pyttsx3 openai-whisper sounddevice
```

> `openai-whisper` pulls `torch` (~2 GB on first install) — be patient.

### Clone this repository

```bash
# 1. Clone
git clone https://github.com/ayerna/RNTBCI-POC-6-demo.git
cd RNTBCI-POC-6-demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the LLM (needs Ollama installed — https://ollama.com)
ollama pull llama3

# 4. Run — the real dataset is already included in the repo
python main.py

# (Optional) Generate a fresh synthetic dataset instead of the real one
# python generate_sample_data.py
```

---

## Running the System

```bash
cd RNTBCI-POC-6-demo
python main.py
```

### Startup sequence

```
─────────────── EV BMS Real-Time Pipeline ──────────────────

Enter your name (or press Enter for default): Gladwin
  Welcome, Gladwin. BMS assistant will personalise responses for you.

[SPEECH] TTS engine (pyttsx3) ready.
[SPEECH] Loading Whisper model 'base'...

  Select Injection Mode:
  [1]  Auto Streaming    — full trip injected automatically
  [2]  Step-by-Step      — press Enter to inject one row at a time
  [3]  Scenario Inject   — inject custom anomaly for instant demo

Mode [1/2/3] > _
```

### Fast validation (no LLM, no speech)

```bash
python smoke_test.py
```

---

## Injection Modes

### Mode 1 — Auto Streaming
- Loads trip data using the `TRIP_START_MODE` strategy (default: `"discharge"` — finds first heavy-discharge segment)
- Streams all `TRIP_ROWS` (default: 7 200) into the buffer in a background thread
- Window worker runs in parallel, generating KB entries continuously
- CLI opens once 5 windows are ready

### Mode 2 — Step-by-Step
```
  [Enter to inject / q to stop] >
```
Each press injects one real row from the dataset. Windows are processed and logged immediately when 30 rows accumulate. Ideal for **live demos** where you want to show the pipeline reacting in real time.

### Mode 3 — Scenario Injection
Choose from 5 built-in scenarios:

| # | Name | What it simulates |
|---|---|---|
| 1 | Thermal Spike | 55°C + 200 A — thermal event risk |
| 2 | Critical Low SOC | SoC 12% — charge immediately |
| 3 | Aggressive Acceleration | 450 A current spike + voltage sag |
| 4 | Regen Braking Recovery | −120 A — strong energy recovery |
| 5 | Custom | Enter your own values |

Each scenario injects 30 identical rows (one full window) so the KB entry appears **immediately** in the log.

---

## Query Interface

Once injection starts and the KB has data, the interactive CLI opens:

```
  [T]  Type your query
  [S]  Speak your query (microphone)
  [D]  Run all demo queries automatically
  [K]  Show knowledge base summary
  [Q]  Quit

  Shortcuts (type a number 1-7):
  1. How is my battery performing so far?
  2. Any issues in the last 30 minutes?
  3. Am I driving aggressively?
  4. What is happening right now?
  5. Is it safe to continue driving?
  6. Show me any critical events during this trip.
  7. What is my current state of charge trend?

You > _
```

### Response format (LLM)
```
Gladwin, Status: Normal. Battery has been operating within safe parameters
across all 24 windows — no warnings or critical events detected this trip.
```

### Response format (Rule Engine fallback)
```
Gladwin, There have been 3 warning window(s) out of 48 analysed (6% of trip
time). High current draw was detected — aggressive acceleration phases occurred.
(Note: This response was generated by the rule engine — LLM was unavailable.)
```

---

## Speech Integration

### Text-to-Speech (TTS)
- Library: `pyttsx3` (fully offline, no API)
- Engine initialised **once** at startup
- Every LLM/fallback answer is spoken aloud automatically
- Long responses truncated to 600 characters for speech
- Prefers female voice (Zira on Windows) if available

### Speech-to-Text (STT)
- Library: `openai-whisper` (`base` model, local inference)
- Records 6 seconds from the default microphone
- Transcription uses `language="en"`, `fp16=False` (CPU-safe)
- Press `[S]` in the CLI to speak your query

### Disabling speech
Set in `config.py`:
```python
SPEECH_ENABLED = False
```

---

## Knowledge Base Format

Each window produces one KB entry:

```json
{
  "window_id": 12,
  "time_range": "2020-01-23 01:24:30+00:00 -> 2020-01-23 01:24:59+00:00",
  "n_rows": 30,
  "observations": [
    "Battery is experiencing high average discharge current — heavy acceleration or sustained high speed",
    "Sudden current spike detected — sharp acceleration burst observed",
    "SoC is depleting faster than expected — possible high load or aggressive driving",
    "Temperature is trending upward — monitor carefully if load continues",
    "Drive mode is predominantly discharging — active driving during this window"
  ],
  "flags": ["HIGH_CURRENT", "CURRENT_SPIKE", "RAPID_SOC_DROP"],
  "inference": "High-load aggressive driving detected. The battery is being discharged aggressively, depleting range faster than normal.",
  "severity": "Warning",
  "metrics": {
    "avg_voltage_V": 401.5,
    "min_voltage_V": 385.2,
    "avg_current_A": 162.3,
    "max_current_A": 453.1,
    "avg_soc_pct": 54.2,
    "min_soc_pct": 53.6,
    "avg_temp_C": 34.1,
    "max_temp_C": 36.8,
    "avg_power_kW": 65.2,
    "soc_rate_pct_s": -0.021,
    "dominant_state": "Discharge"
  },
  "created_at": "2026-05-02T23:15:44.123456"
}
```

The full KB is saved to `knowledge_base.json` after every new window.

---

## LLM Integration & Fallback

### Ollama setup
```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve          # starts on localhost:11434
```

### Query flow
```
User query
    │
    ▼
Fast-path check (simple keyword + no warnings in KB)?
    │ YES → return template answer immediately (no LLM call)
    │ NO
    ▼
Build context (KB overview + last 15 windows + live reading)
    │
    ▼
Ollama llama3 [thread, 45 s timeout]
    │ success → personalised 2-sentence reply
    │ timeout/error
    ▼
Rule-engine fallback (flag-based plain-English summary)
```

### LLM options
```python
options = {
    "num_predict": 60,    # ~2 sentences
    "temperature": 0.2,   # deterministic, no hallucination
}
```

---

## Configuration Reference

**`config.py`** — all values can be changed without touching any other file.

```python
# ── Trip ──────────────────────────────────────────────────────
CSV_PATH        = r"d:\renaults_z\bms_processed_1hz.csv"
TRIP_HOURS      = 2               # 2-hour simulated trip
TRIP_ROWS       = TRIP_HOURS * 3600
TRIP_START_MODE = "discharge"     # "start" | "discharge" | "random"
STREAM_DELAY    = 0.0             # 0 = fast-sim, 1.0 = real-time 1 Hz

# ── Window ────────────────────────────────────────────────────
WINDOW_SIZE     = 30              # rows (= seconds at 1 Hz)
BUFFER_MAX      = 3600            # rolling buffer (1 hour)

# ── LLM ──────────────────────────────────────────────────────
OLLAMA_MODEL    = "llama3:latest"
LLM_TIMEOUT_S   = 45
LLM_MAX_TOKENS  = 60
LLM_TEMPERATURE = 0.2
DRIVER_NAME     = "Driver"        # overridden at startup

# ── Speech ────────────────────────────────────────────────────
WHISPER_MODEL   = "base"          # tiny | base | small | medium
MIC_RECORD_SECS = 6
TTS_RATE        = 165             # words per minute
TTS_VOLUME      = 1.0
SPEECH_ENABLED  = True

# ── Thresholds ────────────────────────────────────────────────
THRESHOLDS = {
    "high_temp":       35.0,   # deg C
    "critical_temp":   37.0,
    "high_current":   150.0,   # A
    "regen_current":  -50.0,
    "low_soc":         25.0,   # %
    "critical_soc":    18.0,
    "rapid_soc_drop":  -0.5,   # %/s
    "high_power":   80_000.0,  # W
    "low_voltage":    370.0,   # V
    "high_voltage":   445.0,
    "current_spike":  400.0,   # A
}
```

---

## Sample Output

### Startup & injection log
```
[SYSTEM] Loading dataset: d:\renaults_z\bms_processed_1hz.csv
+─────────────────── EV Trip Simulation Started ────────────────────+
│  Trip Mode   : discharge                                           │
│  Start       : 2020-01-23 00:24:30+00:00                          │
│  End         : 2020-01-23 02:24:29+00:00                          │
│  Total rows  : 7200 (1 Hz = seconds)                              │
│  Duration    : 2h 0m                                               │
+────────────────────────────────────────────────────────────────────+

[INJECT #   10]  V=  388.50V  I=   24.30A  SoC= 36.4%  T= 17.0C  Discharge
[INJECT #   20]  V=  389.00V  I=   19.80A  SoC= 36.3%  T= 17.0C  Discharge
[WINDOW   1]  Created from 30 rows  (00:24:30 -> 00:24:59)
[KB W  1 | Normal]  Battery is operating within expected parameters...
[WINDOW   2]  Created from 30 rows  (00:25:00 -> 00:25:29)
[KB W  2 | Warning]  High-load aggressive driving detected...
```

### LLM response (first query — greeting)
```
╭─────────────────── BMS Assistant (LLM) ────────────────────╮
│                                                             │
│  Hello Gladwin, Status: Warning. The battery has shown 3   │
│  warning windows with high discharge current and rapid SoC  │
│  depletion — consider easing acceleration to preserve range.│
│                                                             │
╰─────────────────────────────────────────────────────────────╯
```

### LLM response (subsequent queries — personalised)
```
╭─────────────────── BMS Assistant (LLM) ────────────────────╮
│                                                             │
│  Gladwin, Status: Normal. All recent windows are within     │
│  safe parameters and the battery temperature is stable.     │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
```

### Scenario injection (Mode 3 — Thermal Spike)
```
  Injecting scenario: Thermal Spike
[INJECT #    1]  V=  390.00V  I=  200.00A  SoC= 60.0%  T= 55.0C  Discharge
...
[WINDOW   1]  Created from 30 rows
[KB W  1 | CRITICAL]  Thermal event is developing inside the pack. Reduce speed...
  Scenario injected (30 rows). KB updated.
```

### Trip summary table
```
          Trip Knowledge Base Summary
+-----------------------------------------+
│ Metric          │ Value                 │
│─────────────────┼───────────────────────│
│ Total Windows   │ 240                   │
│ CRITICAL Events │ 0                     │
│ Warnings        │ 18                    │
│ Normal Windows  │ 204                   │
│ Info Windows    │ 18                    │
│ Trip Start      │ 2020-01-23 00:24:30   │
│ Last Window     │ 2020-01-23 02:24:29   │
+-----------------------------------------+
```

---

## File Structure

```
RNTBCI-POC-6-demo/
│
├── config.py           ← All constants, thresholds, speech config
├── data_injector.py    ← CSV loader, trip slicer, deque buffer, streaming thread
├── windowing.py        ← Window stats engine + KB entry generator (expert rules)
├── kb_store.py         ← Thread-safe KB storage + JSON persistence + fallback summary
├── llm_agent.py        ← Session, greeting, context builder, Ollama query, fast-path
├── speech.py           ← pyttsx3 TTS + Whisper STT (graceful degradation)
├── logger.py           ← Rich-based colour console for all layers
├── main.py             ← Orchestrator: mode menu, threads, CLI loop
├── smoke_test.py       ← Fast non-interactive validation
├── knowledge_base.json ← Auto-generated KB output (gitignored if large)
└── README.md           ← This file
```

---

## Development History

| Phase | What was built |
|---|---|
| **v1 — Foundation** | CSV injector, deque buffer, fixed-size windows, basic KB generator, Ollama LLM query, CLI loop |
| **v2 — Intelligence** | Natural-language observations (no raw numbers in text), 12-flag expert rule engine, severity inference, 3 trip-selection strategies, rich console logging, pyttsx3 TTS, Whisper STT, Mode 2 step-by-step, Mode 3 scenario injection with 5 presets |
| **v3 — Polish** | Session object (driver name + greeted flag), one-time personalised greeting, strict 2-sentence LLM format, `num_predict=60` / `temperature=0.2`, fast-path rule check (skips LLM for simple queries), LLM thread timeout → automatic fallback, compact context builder |

---

## Requirements

```
pandas
numpy
rich
ollama
pyttsx3
openai-whisper
sounddevice
```

Install all:
```bash
pip install pandas numpy rich ollama pyttsx3 openai-whisper sounddevice
```

---

## License

MIT — see LICENSE file.

---

*Built for RNTBCI POC-6 EV BMS demonstration.*
