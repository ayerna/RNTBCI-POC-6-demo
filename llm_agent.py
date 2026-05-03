"""
llm_agent.py - Layer 6
Context builder + Ollama query with:
  - Session-aware driver greeting (once only)
  - Strict 2-sentence response format
  - Fast rule-based path (skips LLM for trivial state queries)
  - Thread-based timeout + rule-engine fallback
"""

import threading
import ollama
import kb_store
import data_injector
import logger
from config import OLLAMA_MODEL, LLM_TIMEOUT_S, LLM_MAX_TOKENS, LLM_TEMPERATURE


# ─── Session Object ────────────────────────────────────────────────────────────
_session: dict = {
    "driver_name": "Driver",   # set by set_driver_name() at startup
    "greeted":     False,       # becomes True after first LLM response
}


def set_driver_name(name: str) -> None:
    _session["driver_name"] = name.strip() or "Driver"


def get_driver_name() -> str:
    return _session["driver_name"]


# ─── System Prompt Builder ────────────────────────────────────────────────────

def _build_system_prompt(is_first: bool, name: str) -> str:
    greeting_rule = (
        f'Start your reply with "Hello {name},"'
        if is_first
        else f'Start every reply with "{name},"'
    )

    return f"""You are a certified BMS (Battery Management System) safety engineer and EV trip advisor.

{greeting_rule}

Strict response rules:
- EXACTLY 2 sentences maximum. No exceptions.
- FORMAT: {name}, Status: <Normal/Warning/Critical>. Reason: <one concise explanation>.
- Answer ONLY from the provided Knowledge Base and live telemetry. Do NOT hallucinate.
- If a CRITICAL flag exists, lead with it.
- Use plain language a driver understands while driving.
- Never add preamble, lists, or explanations beyond 2 sentences.
"""


# ─── Fast Rule-Based Path (skip LLM for simple queries) ───────────────────────

_SIMPLE_KEYWORDS = {
    "soc":          lambda s: s.get("warning_count", 0) == 0,
    "temperature":  lambda s: s.get("critical_count", 0) == 0,
    "safe":         lambda s: s.get("critical_count", 0) == 0,
    "okay":         lambda s: s.get("critical_count", 0) == 0,
    "fine":         lambda s: s.get("critical_count", 0) == 0,
}


def _try_fast_path(query: str) -> str | None:
    """
    If the query matches a simple keyword AND the KB shows no critical/warning
    events, return a quick templated answer without hitting the LLM.
    Returns None if fast path is not applicable.
    """
    stats = kb_store.summary_stats()
    if not stats:
        return None

    name    = _session["driver_name"]
    is_first = not _session["greeted"]
    q_low   = query.lower()

    for kw, is_simple_fn in _SIMPLE_KEYWORDS.items():
        if kw in q_low and is_simple_fn(stats):
            total  = stats["total_windows"]
            prefix = f"Hello {name}," if is_first else f"{name},"
            return (
                f"{prefix} Status: Normal. "
                f"All {total} analysed windows show no critical or warning conditions — "
                f"the battery is operating within safe parameters."
            )
    return None


# ─── Context Builder ──────────────────────────────────────────────────────────

def _build_context(max_kb_entries: int = 15) -> str:
    stats     = kb_store.summary_stats()
    criticals = kb_store.get_by_severity("CRITICAL")
    recent    = kb_store.get_recent(max_kb_entries)

    if not stats:
        kb_overview = "Knowledge Base: No windows processed yet.\n"
    else:
        kb_overview = (
            f"=== TRIP KB OVERVIEW ===\n"
            f"  Windows analysed : {stats['total_windows']}\n"
            f"  CRITICAL         : {stats['critical_count']}\n"
            f"  Warnings         : {stats['warning_count']}\n"
            f"  Normal           : {stats['normal_count']}\n\n"
        )

    def _fmt(e: dict) -> str:
        obs = "; ".join(e.get("observations", [])[:4])   # cap observations for brevity
        m   = e.get("metrics", {})
        return (
            f"[W{e['window_id']} | {e['severity']}] {obs}\n"
            f"  Inference: {e.get('inference','')}\n"
            f"  V={m.get('avg_voltage_V')}V  I={m.get('avg_current_A')}A  "
            f"SoC={m.get('avg_soc_pct')}%  T={m.get('avg_temp_C')}C\n"
        )

    recent_block   = "=== RECENT WINDOWS ===\n" + "".join(_fmt(e) for e in recent)
    critical_block = ""
    if criticals:
        critical_block = "\n=== CRITICAL EVENTS ===\n" + "".join(_fmt(e) for e in criticals[-3:])

    live_rows = data_injector.get_latest(3)
    if live_rows:
        lr = live_rows[-1]
        live_block = (
            "\n=== LIVE READING ===\n"
            f"  V={lr.get('voltage','?')}V  I={lr.get('current','?')}A  "
            f"SoC={lr.get('soc','?')}%  T={lr.get('temperature','?')}C  "
            f"State={lr.get('state','?')}\n"
        )
    else:
        live_block = "\n=== LIVE READING ===\nNo data yet.\n"

    return kb_overview + recent_block + critical_block + live_block


# ─── LLM Query ────────────────────────────────────────────────────────────────

def query_llm(user_query: str, verbose: bool = True) -> tuple[str, str]:
    """
    Returns (answer: str, source: str) where source = "LLM" | "FastPath" | "RuleEngine"
    """
    name = _session["driver_name"]

    # ── Fast path check ───────────────────────────────────────────────────────
    fast = _try_fast_path(user_query)
    if fast:
        if not _session["greeted"]:
            _session["greeted"] = True
        if verbose:
            logger.log_llm("Fast-path answer (no LLM call needed).")
        return fast, "FastPath"

    # ── Build messages ────────────────────────────────────────────────────────
    is_first = not _session["greeted"]
    system_p = _build_system_prompt(is_first, name)
    context  = _build_context()

    kb_size = len(kb_store.get_all())
    if verbose:
        logger.log_llm(
            f"Querying {OLLAMA_MODEL} | KB={kb_size} windows | "
            f"first_greeting={is_first} | tokens_max={LLM_MAX_TOKENS}"
        )

    messages = [
        {"role": "system", "content": system_p},
        {"role": "user",   "content": f"{context}\n\nDriver query: {user_query}"},
    ]

    # ── Thread + timeout ─────────────────────────────────────────────────────
    result_holder: list[str] = []
    error_holder:  list[str] = []

    def _call():
        try:
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={
                    "num_predict": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                },
            )
            result_holder.append(resp["message"]["content"].strip())
        except Exception as e:
            error_holder.append(str(e))

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=LLM_TIMEOUT_S)

    if result_holder:
        _session["greeted"] = True
        if verbose:
            logger.log_llm("Response received.")
        return result_holder[0], "LLM"

    # ── Fallback ─────────────────────────────────────────────────────────────
    reason = error_holder[0] if error_holder else f"Timeout after {LLM_TIMEOUT_S}s"
    if verbose:
        logger.log_llm(f"LLM unavailable ({reason}) — rule engine fallback.")
    fallback = kb_store.rule_based_summary(user_query)
    # Personalise fallback with name
    if not fallback.startswith(name):
        fallback = f"{name}, " + fallback
    _session["greeted"] = True
    return fallback, "RuleEngine"
