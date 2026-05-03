"""
main.py - BMS Real-Time Pipeline Orchestrator
=============================================
Layers:
  1/2 - Data injection  (data_injector)
  3   - Windowing       (windowing)
  4   - KB generation   (windowing)
  5   - KB storage      (kb_store)
  6   - LLM / Fallback  (llm_agent)
  +   - Speech I/O      (speech)
  +   - Rich logging    (logger)

Injection modes:
  1 - Auto Streaming     (background thread, full trip)
  2 - Step-by-Step       (manual Enter key per row)
  3 - Scenario Inject    (inject a custom anomaly dict)
"""

import sys
import time
import threading
from datetime import datetime

import data_injector
import kb_store
import llm_agent
import speech as speech_mod
import logger
from windowing import process_window, generate_kb_entry
from config import WINDOW_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# SHARED WINDOWING STATE  (used by all injection modes)
# ─────────────────────────────────────────────────────────────────────────────
_window_id_counter = 1
_processed_idx     = 0
_win_lock          = threading.Lock()


def _process_ready_windows() -> None:
    """Process any complete windows sitting in the buffer. Call after each inject."""
    global _window_id_counter, _processed_idx
    with _win_lock:
        snapshot  = data_injector.get_buffer_snapshot()
        available = len(snapshot)
        while available - _processed_idx >= WINDOW_SIZE:
            chunk   = snapshot[_processed_idx : _processed_idx + WINDOW_SIZE]
            summary = process_window(chunk)
            if summary:
                entry = generate_kb_entry(_window_id_counter, summary)
                kb_store.add_entry(entry)
                _window_id_counter += 1
            _processed_idx += WINDOW_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# ROW CALLBACK  (mode 1 — every 10th row)
# ─────────────────────────────────────────────────────────────────────────────

def _on_row(record: dict, idx: int) -> None:
    if idx % 10 != 0:
        return
    logger.log_inject(
        row_idx = idx,
        voltage = record.get("voltage", 0),
        current = record.get("current", 0),
        soc     = record.get("soc", 0),
        temp    = record.get("temperature", 0),
        state   = record.get("state", "?"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWING WORKER  (mode 1 background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _window_worker() -> None:
    global _window_id_counter, _processed_idx
    while True:
        _process_ready_windows()
        if data_injector.injection_complete():
            # flush partial tail window
            with _win_lock:
                remaining = data_injector.get_buffer_snapshot()[_processed_idx:]
                if len(remaining) >= 5:
                    summary = process_window(remaining)
                    if summary:
                        entry = generate_kb_entry(_window_id_counter, summary)
                        kb_store.add_entry(entry)
                        _window_id_counter += 1
            break
        time.sleep(0.4)
    logger.log_system(f"Windowing complete. {len(kb_store.get_all())} KB entries generated.")


# ─────────────────────────────────────────────────────────────────────────────
# INJECTION MODE 2 — Step-by-Step
# ─────────────────────────────────────────────────────────────────────────────

def _mode_step_by_step(trip_df, tts_ok: bool, stt_ok: bool) -> None:
    from logger import console
    from rich.panel import Panel
    from rich import box

    console.print(Panel(
        "  Press [bold]Enter[/bold] to inject one data point at a time.\n"
        "  Type [bold]q[/bold] + Enter to stop and open the query interface.",
        title="[window]Mode 2 - Step-by-Step Injection[/window]",
        border_style="bright_blue", box=box.ROUNDED
    ))

    rows = list(trip_df.iterrows())
    injected = 0

    for _, row in rows:
        try:
            raw = input("  [Enter to inject / q to stop] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if raw == "q":
            break

        record = {
            "timestamp":   str(row.get("timestamp", datetime.now().isoformat())),
            "voltage":     float(row.get("voltage",     0)),
            "current":     float(row.get("current",     0)),
            "soc":         float(row.get("soc",         0)),
            "temperature": float(row.get("temperature", 0)),
            "power_w":     float(row.get("power_w",     0)),
            "state":       str(row.get("state",         "Unknown")),
            "injected_at": datetime.now().isoformat(),
        }
        from collections import deque
        import data_injector as _di
        with _di._buffer_lock:
            _di._buffer.append(record)
        _di._row_count += 1
        injected += 1

        logger.log_inject(
            row_idx = injected,
            voltage = record["voltage"],
            current = record["current"],
            soc     = record["soc"],
            temp    = record["temperature"],
            state   = record["state"],
        )
        _process_ready_windows()

    data_injector._inject_done.set()
    logger.log_system(f"Step-by-step done. {injected} rows injected.")
    _cli_loop(tts_ok, stt_ok)


# ─────────────────────────────────────────────────────────────────────────────
# INJECTION MODE 3 — Scenario (anomaly) Injection
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "1": {
        "name":  "Thermal Spike",
        "desc":  "Temperature 55C + high current — simulates thermal event",
        "data":  {"voltage": 390.0, "current": 200.0, "temperature": 55.0,
                  "soc": 60.0, "power_w": 78000.0, "state": "Discharge"},
    },
    "2": {
        "name":  "Critical Low SOC",
        "desc":  "SOC drops to 12% — low charge critical state",
        "data":  {"voltage": 371.0, "current": 45.0, "temperature": 28.0,
                  "soc": 12.0, "power_w": 16650.0, "state": "Discharge"},
    },
    "3": {
        "name":  "Aggressive Acceleration",
        "desc":  "Current spike 450A + rapid voltage sag",
        "data":  {"voltage": 380.0, "current": 450.0, "temperature": 36.0,
                  "soc": 55.0, "power_w": 171000.0, "state": "Discharge"},
    },
    "4": {
        "name":  "Regen Braking Recovery",
        "desc":  "Negative current -120A — strong energy recovery",
        "data":  {"voltage": 430.0, "current": -120.0, "temperature": 22.0,
                  "soc": 65.0, "power_w": -51600.0, "state": "Regen"},
    },
    "5": {
        "name":  "Custom",
        "desc":  "Enter your own values",
        "data":  None,
    },
}


def _inject_single_record(data: dict) -> None:
    """Push one synthetic record into the buffer and process windows."""
    record = {
        "timestamp":   datetime.now().isoformat(),
        "voltage":     data.get("voltage",     400.0),
        "current":     data.get("current",     10.0),
        "soc":         data.get("soc",         60.0),
        "temperature": data.get("temperature", 25.0),
        "power_w":     data.get("power_w",     4000.0),
        "state":       data.get("state",       "Discharge"),
        "injected_at": datetime.now().isoformat(),
    }
    import data_injector as _di
    with _di._buffer_lock:
        _di._buffer.append(record)
    _di._row_count += 1

    logger.log_inject(
        row_idx = _di._row_count,
        voltage = record["voltage"],
        current = record["current"],
        soc     = record["soc"],
        temp    = record["temperature"],
        state   = record["state"],
    )


def _mode_scenario(tts_ok: bool, stt_ok: bool) -> None:
    from logger import console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    # Show scenario menu
    t = Table(title="Available Scenarios", box=box.ROUNDED, border_style="bright_blue")
    t.add_column("#",       style="bold cyan", width=4)
    t.add_column("Name",    style="bold white")
    t.add_column("Description", style="dim")
    for k, sc in SCENARIOS.items():
        t.add_row(k, sc["name"], sc["desc"])
    console.print(t)

    while True:
        try:
            choice = console.input(
                "[bold magenta]Select scenario (1-5) or [q]uit to query interface > [/bold magenta]"
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q":
            break

        sc = SCENARIOS.get(choice)
        if not sc:
            console.print("  [yellow]Invalid choice.[/yellow]")
            continue

        if sc["data"] is None:
            # Custom entry
            try:
                v = float(console.input("  Voltage (V)     : "))
                i = float(console.input("  Current (A)     : "))
                temp = float(console.input("  Temperature (C) : "))
                soc  = float(console.input("  SoC (%)         : "))
            except (ValueError, EOFError):
                console.print("  [red]Invalid input.[/red]")
                continue
            data = {"voltage": v, "current": i, "temperature": temp, "soc": soc,
                    "power_w": v * i, "state": "Discharge"}
        else:
            data = sc["data"]

        console.print(f"\n  [bright_blue]Injecting scenario:[/bright_blue] [bold]{sc['name']}[/bold]")
        # Inject WINDOW_SIZE copies so a full window is immediately available
        for _ in range(WINDOW_SIZE):
            _inject_single_record(data)

        _process_ready_windows()
        console.print(
            f"  [green]Scenario injected ({WINDOW_SIZE} rows). "
            "KB updated — check [KB] log above.[/green]\n"
        )

    data_injector._inject_done.set()
    _cli_loop(tts_ok, stt_ok)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE QUERY CLI
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "How is my battery performing so far?",
    "Any issues in the last 30 minutes?",
    "Am I driving aggressively?",
    "What is happening right now?",
    "Is it safe to continue driving?",
    "Show me any critical events during this trip.",
    "What is my current state of charge trend?",
]


def _show_query_menu(tts_ok: bool, stt_ok: bool) -> None:
    from rich.panel import Panel
    from rich import box
    from logger import console

    lines = [
        "  [T]  Type your query",
        "  [S]  Speak your query (microphone)" if stt_ok else "  [S]  (Voice unavailable)",
        "  [D]  Run all demo queries automatically",
        "  [K]  Show knowledge base summary",
        "  [Q]  Quit",
        "",
        "[dim]Shortcuts (type a number 1-7):[/dim]",
    ] + [f"  {i+1}. {q}" for i, q in enumerate(DEMO_QUERIES)]

    console.print(Panel(
        "\n".join(lines),
        title=f"[system]BMS Assistant — {llm_agent.get_driver_name()}[/system]",
        border_style="bright_blue", box=box.ROUNDED, padding=(1, 2)
    ))
    tts_str = "[green]ON[/green]" if tts_ok else "[red]OFF[/red]"
    stt_str = "[green]ON[/green]" if stt_ok else "[red]OFF[/red]"
    console.print(f"  [dim]TTS: {tts_str}  |  STT: {stt_str}[/dim]\n")


def _handle_query(query: str, tts_ok: bool) -> None:
    answer, source = llm_agent.query_llm(query, verbose=True)
    logger.show_answer(answer, source)
    if tts_ok:
        speech_mod.speak(answer)


def _cli_loop(tts_ok: bool, stt_ok: bool) -> None:
    _show_query_menu(tts_ok, stt_ok)
    from logger import console

    while True:
        try:
            raw = console.input("[bold magenta]You > [/bold magenta]").strip()
        except (EOFError, KeyboardInterrupt):
            logger.log_system("Shutting down.")
            break

        if not raw:
            continue
        low = raw.lower()

        if low in ("q", "quit", "exit"):
            logger.log_system("Goodbye.")
            break

        if raw.isdigit() and 1 <= int(raw) <= len(DEMO_QUERIES):
            query = DEMO_QUERIES[int(raw) - 1]
            console.print(f"  [dim]-> {query}[/dim]")
            _handle_query(query, tts_ok)
            continue

        if low == "t":
            try:
                query = console.input("  [cyan]Type your question: [/cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if query:
                _handle_query(query, tts_ok)
            continue

        if low == "s":
            if not stt_ok:
                console.print("  [yellow]Voice input unavailable — please type.[/yellow]")
                continue
            query = speech_mod.listen()
            if query:
                console.print(f"  [dim]Transcribed: \"{query}\"[/dim]")
                _handle_query(query, tts_ok)
            else:
                console.print("  [yellow]No speech detected. Try again.[/yellow]")
            continue

        if low == "d":
            console.print("\n[dim]Running demo queries...[/dim]\n")
            for q in DEMO_QUERIES:
                console.print(f"\n[dim]-> {q}[/dim]")
                _handle_query(q, tts_ok)
                time.sleep(0.5)
            continue

        if low == "k":
            stats = kb_store.summary_stats()
            if stats:
                logger.show_summary_table(stats)
            else:
                console.print("[yellow]KB empty — waiting for data.[/yellow]")
            continue

        if len(raw) > 3:
            _handle_query(raw, tts_ok)
        else:
            console.print("  [dim]Type T / S / D / K / Q, a number, or a question.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTION MENU
# ─────────────────────────────────────────────────────────────────────────────

def _show_mode_menu() -> None:
    from rich.panel import Panel
    from rich import box
    from logger import console

    body = (
        "  [bold cyan][1][/bold cyan]  Auto Streaming    — full trip injected automatically\n"
        "  [bold cyan][2][/bold cyan]  Step-by-Step      — press Enter to inject one row at a time\n"
        "  [bold cyan][3][/bold cyan]  Scenario Inject   — inject custom anomaly for instant demo"
    )
    console.print(Panel(body, title="[system]Select Injection Mode[/system]",
                        border_style="bright_blue", box=box.DOUBLE_EDGE, padding=(1, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from logger import console
    from rich.rule import Rule

    console.print(Rule("[bold bright_blue]EV BMS Real-Time Pipeline[/bold bright_blue]"))

    # ── 1. Driver name (session personalisation) ──────────────────────────────
    try:
        name = console.input(
            "[bold cyan]Enter your name (or press Enter for default): [/bold cyan]"
        ).strip()
    except (EOFError, KeyboardInterrupt):
        name = ""
    llm_agent.set_driver_name(name or "Driver")
    driver_name = llm_agent.get_driver_name()
    console.print(
        f"  [dim]Welcome, [bold]{driver_name}[/bold]. "
        "BMS assistant will personalise responses for you.[/dim]\n"
    )

    # ── 2. Initialise speech ──────────────────────────────────────────────────
    tts_ok, stt_ok = speech_mod.init_speech()

    # Speak the personalised welcome greeting (non-blocking so menu appears immediately)
    if tts_ok:
        speech_mod.speak(f"Hello {driver_name}, welcome. I am your BMS assistant. I will keep you informed about your battery health during the trip.")

    # ── 3. Mode selection ─────────────────────────────────────────────────────
    _show_mode_menu()
    try:
        mode = console.input("[bold magenta]Mode [1/2/3] > [/bold magenta]").strip()
    except (EOFError, KeyboardInterrupt):
        mode = "1"
    if mode not in ("1", "2", "3"):
        mode = "1"

    # ── 4. Load trip data (all modes need it) ─────────────────────────────────
    trip_df = data_injector.load_trip_data()

    # ─────────────────────────────────────────────────────────────────────────
    if mode == "1":
        # Auto streaming
        inj_thread = data_injector.start_injection(trip_df, on_row_callback=_on_row)
        win_thread = threading.Thread(target=_window_worker, daemon=True, name="WindowWorker")
        win_thread.start()
        logger.log_system(f"Mode 1 — Auto streaming started (window = {WINDOW_SIZE} rows).")

        # Wait for 5 windows
        logger.log_system("Pre-loading KB — waiting for 5 windows...")
        spinner = ["|", "/", "-", "\\"]
        tick = 0
        while len(kb_store.get_all()) < 5 and not data_injector.injection_complete():
            console.print(f"  [dim]{spinner[tick % 4]} buffering...[/dim]", end="\r")
            tick += 1
            time.sleep(0.3)
        console.print(" " * 30, end="\r")
        logger.log_system(f"KB ready ({len(kb_store.get_all())} windows).")

        _cli_loop(tts_ok, stt_ok)
        inj_thread.join(timeout=5)
        win_thread.join(timeout=10)

    elif mode == "2":
        # Step-by-step
        logger.log_system("Mode 2 — Step-by-step injection.")
        _mode_step_by_step(trip_df, tts_ok, stt_ok)

    elif mode == "3":
        # Scenario injection — no trip data needed for the scenarios themselves
        logger.log_system("Mode 3 — Scenario injection.")
        data_injector._inject_done.clear()   # keep door open
        _mode_scenario(tts_ok, stt_ok)

    # ── Final summary ─────────────────────────────────────────────────────────
    stats = kb_store.summary_stats()
    if stats:
        console.print()
        logger.show_summary_table(stats)
        console.print(f"\n[dim]KB saved to: {__import__('config').KB_JSON_PATH}[/dim]")

    console.print(Rule("[bold bright_blue]Session ended[/bold bright_blue]"))


if __name__ == "__main__":
    main()
