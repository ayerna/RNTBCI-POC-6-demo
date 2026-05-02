"""
smoke_test_v2.py
Fast validation: inject 300 rows, process windows, check KB quality.
No LLM or speech required.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Patch config for speed
import config
config.TRIP_ROWS        = 300
config.STREAM_DELAY     = 0.0
config.WINDOW_SIZE      = 30
config.TRIP_START_MODE  = "discharge"
config.SPEECH_ENABLED   = False

import time
import data_injector
import kb_store
from windowing import process_window, generate_kb_entry
from logger import console

console.print("\n[bold bright_blue]BMS Pipeline v2 - Smoke Test[/bold bright_blue]\n")

# Load & inject
trip_df = data_injector.load_trip_data()
inj = data_injector.start_injection(trip_df)
inj.join(timeout=10)

console.print(f"[green]Rows injected: {data_injector.get_row_count()}[/green]")

buf = data_injector.get_buffer_snapshot()
console.print(f"[green]Buffer size  : {len(buf)}[/green]\n")

# Build windows
wid = 1
for i in range(0, len(buf) - config.WINDOW_SIZE + 1, config.WINDOW_SIZE):
    chunk   = buf[i : i + config.WINDOW_SIZE]
    summary = process_window(chunk)
    entry   = generate_kb_entry(wid, summary)
    kb_store.add_entry(entry)
    wid += 1

# Show full KB entry for last window
from logger import show_kb_entry, show_summary_table
last = kb_store.get_recent(1)[0]
console.print("\n[bold]Full KB entry (last window):[/bold]")
show_kb_entry(last)

# Summary
stats = kb_store.summary_stats()
show_summary_table(stats)

# Test rule-based fallback
fb = kb_store.rule_based_summary("Am I driving aggressively?")
console.print(f"\n[magenta][FALLBACK ANSWER][/magenta] {fb}\n")

console.print("[bold green]Smoke test v2 PASSED[/bold green]\n")
