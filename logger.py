"""
logger.py - Centralised Rich console logging for BMS pipeline.
All modules import from here so styling is consistent.
"""

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich import box
import datetime

_theme = Theme({
    "inject":  "cyan",
    "window":  "bright_blue bold",
    "kb.normal":   "green",
    "kb.info":     "bright_cyan",
    "kb.warning":  "yellow bold",
    "kb.critical": "red bold",
    "llm":     "magenta",
    "speech":  "bright_magenta",
    "system":  "bright_white bold",
    "dim":     "grey50",
})

console = Console(theme=_theme, highlight=False)


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


# ── Per-layer loggers ──────────────────────────────────────────────────────────

def log_inject(row_idx: int, voltage: float, current: float,
               soc: float, temp: float, state: str) -> None:
    console.print(
        f"[dim][{_ts()}][/dim] [inject][INJECT #{row_idx:>5}][/inject]  "
        f"V=[bold]{voltage:>7.2f}[/bold]V  "
        f"I=[bold]{current:>8.2f}[/bold]A  "
        f"SoC=[bold]{soc:>5.1f}[/bold]%  "
        f"T=[bold]{temp:>5.1f}[/bold]C  "
        f"[dim]{state}[/dim]"
    )


def log_window(window_id: int, n_rows: int, time_range: str) -> None:
    console.print(
        f"[dim][{_ts()}][/dim] [window][WINDOW {window_id:>3}][/window]  "
        f"Created from {n_rows} rows  [dim]({time_range})[/dim]"
    )


def log_kb(window_id: int, severity: str, inference: str) -> None:
    sev_tag = {
        "CRITICAL": "kb.critical",
        "Warning":  "kb.warning",
        "Normal":   "kb.normal",
        "Info":     "kb.info",
    }.get(severity, "kb.normal")

    console.print(
        f"[dim][{_ts()}][/dim] [{sev_tag}][KB W{window_id:>3} | {severity:>8}][/{sev_tag}]  "
        f"{inference[:90]}{'...' if len(inference) > 90 else ''}"
    )


def log_llm(msg: str) -> None:
    console.print(f"[dim][{_ts()}][/dim] [llm][LLM][/llm] {msg}")


def log_speech(msg: str) -> None:
    console.print(f"[dim][{_ts()}][/dim] [speech][SPEECH][/speech] {msg}")


def log_system(msg: str) -> None:
    console.print(f"[dim][{_ts()}][/dim] [system][SYSTEM][/system] {msg}")


# ── Rich panels for big display moments ───────────────────────────────────────

def show_trip_banner(start_time: str, end_time: str, n_rows: int, mode: str) -> None:
    body = Text.assemble(
        ("  Trip Mode   : ", "dim"), (mode, "bright_white"), "\n",
        ("  Start       : ", "dim"), (start_time, "bright_white"), "\n",
        ("  End         : ", "dim"), (end_time, "bright_white"), "\n",
        ("  Total rows  : ", "dim"), (f"{n_rows:,}", "bright_white"), " (1 Hz = seconds)\n",
        ("  Duration    : ", "dim"), (f"{n_rows // 3600}h {(n_rows % 3600) // 60}m", "bright_white"),
    )
    console.print(Panel(body, title="[system]EV Trip Simulation Started[/system]",
                        border_style="bright_blue", box=box.DOUBLE_EDGE, padding=(1, 2)))


def show_answer(answer: str, source: str = "LLM") -> None:
    console.print(Panel(
        answer,
        title=f"[llm]BMS Assistant ({source})[/llm]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def show_kb_entry(entry: dict) -> None:
    """Full pretty-print of one KB entry."""
    sev = entry.get("severity", "Normal")
    color = {"CRITICAL": "red", "Warning": "yellow", "Normal": "green", "Info": "cyan"}.get(sev, "white")
    lines = []
    for obs in entry.get("observations", []):
        lines.append(f"  - {obs}")
    obs_block = "\n".join(lines)
    body = (
        f"[dim]Time     :[/dim]  {entry.get('time_range', '')}\n"
        f"[dim]Severity :[/dim]  [{color}]{sev}[/{color}]\n"
        f"[dim]Inference:[/dim]  {entry.get('inference', '')}\n"
        f"[dim]Observations:[/dim]\n{obs_block}\n"
        f"[dim]Metrics  :[/dim]  "
        f"V={entry['metrics']['avg_voltage_V']}V  "
        f"I={entry['metrics']['avg_current_A']}A  "
        f"SoC={entry['metrics']['avg_soc_pct']}%  "
        f"T={entry['metrics']['avg_temp_C']}C  "
        f"P={entry['metrics']['avg_power_kW']}kW"
    )
    console.print(Panel(body,
                        title=f"[{color}]Window {entry['window_id']} Knowledge Base Entry[/{color}]",
                        border_style=color, box=box.SIMPLE_HEAD))


def show_summary_table(stats: dict) -> None:
    from rich.table import Table
    t = Table(title="Trip Knowledge Base Summary", box=box.ROUNDED, border_style="bright_blue")
    t.add_column("Metric",     style="dim",          no_wrap=True)
    t.add_column("Value",      style="bright_white")
    t.add_row("Total Windows",   str(stats.get("total_windows", 0)))
    t.add_row("CRITICAL Events", f"[red bold]{stats.get('critical_count', 0)}[/red bold]")
    t.add_row("Warnings",        f"[yellow]{stats.get('warning_count', 0)}[/yellow]")
    t.add_row("Normal Windows",  f"[green]{stats.get('normal_count', 0)}[/green]")
    t.add_row("Info Windows",    f"[cyan]{stats.get('info_count', 0)}[/cyan]")
    t.add_row("Trip Start",      stats.get("first_window_time", "-"))
    t.add_row("Last Window",     stats.get("last_window_time",  "-"))
    console.print(t)
