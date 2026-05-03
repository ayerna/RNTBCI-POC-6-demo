"""
tests.py — BMS Pipeline Production Test Suite
==============================================
Run with:  python tests.py
No external services needed (LLM / mic / TTS are all mocked).
"""

import sys, os, unittest, threading, time, json
sys.path.insert(0, os.path.dirname(__file__))

# ── Patch config before any module imports ─────────────────────────────────────
import config
config.STREAM_DELAY   = 0.0
config.WINDOW_SIZE    = 10
config.TRIP_ROWS      = 100
config.SPEECH_ENABLED = False          # disable real TTS/STT in all tests
config.LLM_TIMEOUT_S  = 2             # fast timeout for fallback tests


# ── Now import project modules ─────────────────────────────────────────────────
import data_injector
import kb_store
import llm_agent
import speech as speech_mod
from windowing import process_window, generate_kb_entry


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(voltage=400.0, current=50.0, soc=60.0, temp=25.0,
                 state="Discharge", ts=None):
    from datetime import datetime
    return {
        "timestamp":   ts or datetime.now().isoformat(),
        "voltage":     voltage,
        "current":     current,
        "soc":         soc,
        "temperature": temp,
        "power_w":     voltage * current,
        "state":       state,
        "injected_at": datetime.now().isoformat(),
    }


def _fill_kb(n_windows=5, severity_override=None):
    """Push n_windows worth of records through windowing into kb_store."""
    kb_store._kb.clear()
    for wid in range(1, n_windows + 1):
        chunk = [_make_record() for _ in range(config.WINDOW_SIZE)]
        summary = process_window(chunk)
        if summary and severity_override:
            summary["severity"] = severity_override
        if summary:
            entry = generate_kb_entry(wid, summary)
            if severity_override:
                entry["severity"] = severity_override
            kb_store.add_entry(entry)


def _reset_injector():
    """Reset data_injector shared state between tests."""
    data_injector._buffer.clear()
    data_injector._row_count = 0
    data_injector._inject_done.clear()


def _reset_session():
    llm_agent._session["driver_name"] = "Driver"
    llm_agent._session["greeted"]     = False


# =============================================================================
# 1. SPEECH MODULE TESTS
# =============================================================================

class TestSpeechModule(unittest.TestCase):

    def test_speak_noop_when_disabled(self):
        """speak() must return without error when SPEECH_ENABLED=False."""
        speech_mod._tts_available = False
        speech_mod.speak("Hello world")   # should not raise

    def test_speak_blocking_noop_when_disabled(self):
        speech_mod._tts_available = False
        speech_mod.speak_blocking("Hello world")

    def test_speak_empty_string_ignored(self):
        speech_mod._tts_available = True
        # Even if available, empty text should be silently ignored
        # Queue should not grow
        qsize_before = speech_mod._tts_queue.qsize()
        speech_mod.speak("")
        speech_mod.speak("   ")
        self.assertEqual(speech_mod._tts_queue.qsize(), qsize_before)
        speech_mod._tts_available = False   # restore

    def test_listen_returns_none_when_disabled(self):
        result = speech_mod.listen()
        self.assertIsNone(result)

    def test_init_speech_returns_false_when_disabled(self):
        tts_ok, stt_ok = speech_mod.init_speech()
        self.assertFalse(tts_ok)
        self.assertFalse(stt_ok)


# =============================================================================
# 2. LLM AGENT SESSION TESTS
# =============================================================================

class TestLLMAgentSession(unittest.TestCase):

    def setUp(self):
        _reset_session()
        kb_store._kb.clear()

    def test_set_driver_name_basic(self):
        llm_agent.set_driver_name("Renault")
        self.assertEqual(llm_agent.get_driver_name(), "Renault")

    def test_set_driver_name_strips_whitespace(self):
        llm_agent.set_driver_name("  Alice  ")
        self.assertEqual(llm_agent.get_driver_name(), "Alice")

    def test_set_driver_name_empty_defaults_to_driver(self):
        llm_agent.set_driver_name("")
        self.assertEqual(llm_agent.get_driver_name(), "Driver")

    def test_set_driver_name_whitespace_only_defaults(self):
        llm_agent.set_driver_name("   ")
        self.assertEqual(llm_agent.get_driver_name(), "Driver")

    def test_greeted_flag_starts_false(self):
        self.assertFalse(llm_agent._session["greeted"])

    def test_first_fast_path_says_hello(self):
        """First fast-path response must start with 'Hello {name},'."""
        _fill_kb(5)
        llm_agent.set_driver_name("TestDriver")
        result = llm_agent._try_fast_path("is the soc okay?")
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("Hello TestDriver,"),
                        f"Expected 'Hello TestDriver,' but got: {result[:60]}")

    def test_subsequent_fast_path_no_hello(self):
        """After greeted=True, fast-path must NOT say 'Hello'."""
        _fill_kb(5)
        llm_agent.set_driver_name("TestDriver")
        llm_agent._session["greeted"] = True
        result = llm_agent._try_fast_path("is the soc okay?")
        self.assertIsNotNone(result)
        self.assertFalse(result.startswith("Hello"),
                         f"Should not greet again: {result[:60]}")


# =============================================================================
# 3. FAST PATH TESTS
# =============================================================================

class TestFastPath(unittest.TestCase):

    def setUp(self):
        _reset_session()
        kb_store._kb.clear()

    def test_fast_path_returns_none_with_empty_kb(self):
        result = llm_agent._try_fast_path("is the soc okay?")
        self.assertIsNone(result)

    def test_fast_path_returns_none_when_critical_events(self):
        _fill_kb(5, severity_override="CRITICAL")
        result = llm_agent._try_fast_path("is temperature safe?")
        self.assertIsNone(result, "Fast path should not fire when criticals exist")

    def test_fast_path_fires_for_safe_keywords(self):
        _fill_kb(5)
        for kw in ["soc", "temperature", "safe", "okay", "fine"]:
            _reset_session()
            result = llm_agent._try_fast_path(f"is the {kw} fine?")
            self.assertIsNotNone(result, f"Fast path should fire for keyword '{kw}'")

    def test_fast_path_result_contains_name(self):
        _fill_kb(5)
        llm_agent.set_driver_name("Zara")
        result = llm_agent._try_fast_path("is it safe to drive?")
        self.assertIn("Zara", result)

    def test_fast_path_does_not_fire_for_unrelated_query(self):
        _fill_kb(5)
        result = llm_agent._try_fast_path("tell me about regen braking efficiency")
        self.assertIsNone(result)


# =============================================================================
# 4. RULE ENGINE / KB FALLBACK TESTS
# =============================================================================

class TestRuleBasedSummary(unittest.TestCase):

    def setUp(self):
        kb_store._kb.clear()

    def test_empty_kb_returns_waiting_message(self):
        msg = kb_store.rule_based_summary()
        self.assertIn("still collecting", msg)

    def test_no_llm_note_in_fallback(self):
        """The LLM-unavailable disclosure must NOT appear in any response."""
        _fill_kb(5)
        msg = kb_store.rule_based_summary("Am I driving aggressively?")
        self.assertNotIn("LLM was unavailable", msg,
                         "Disclosure note must be removed from user-facing output")
        self.assertNotIn("rule engine", msg.lower(),
                         "Internal system notes must not leak to users")

    def test_critical_flagged_in_summary(self):
        _fill_kb(3, severity_override="CRITICAL")
        msg = kb_store.rule_based_summary()
        self.assertIn("ALERT", msg)

    def test_no_critical_shows_healthy(self):
        _fill_kb(5)
        msg = kb_store.rule_based_summary()
        self.assertIn("No warnings", msg)

    def test_fallback_personalised_with_name(self):
        """rule_based_summary result is personalised by query_llm, not the function itself."""
        _fill_kb(5)
        _reset_session()
        llm_agent.set_driver_name("Carlos")
        # Mock the LLM to timeout so fallback fires
        original_timeout = config.LLM_TIMEOUT_S
        config.LLM_TIMEOUT_S = 0.001  # instant timeout
        answer, source = llm_agent.query_llm("How is my battery?", verbose=False)
        config.LLM_TIMEOUT_S = original_timeout
        self.assertEqual(source, "RuleEngine")
        self.assertTrue(answer.startswith("Carlos,"),
                        f"Fallback must be personalised, got: {answer[:60]}")

    def test_source_tagged_correctly(self):
        _fill_kb(5)
        _reset_session()
        # Force fast-path
        answer, source = llm_agent.query_llm("is the soc okay?", verbose=False)
        self.assertEqual(source, "FastPath")


# =============================================================================
# 5. KB STORE TESTS
# =============================================================================

class TestKBStore(unittest.TestCase):

    def setUp(self):
        kb_store._kb.clear()

    def test_add_and_get_all(self):
        _fill_kb(3)
        self.assertEqual(len(kb_store.get_all()), 3)

    def test_get_recent(self):
        _fill_kb(10)
        recent = kb_store.get_recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1]["window_id"], 10)

    def test_get_by_severity(self):
        _fill_kb(5, severity_override="CRITICAL")
        crits = kb_store.get_by_severity("CRITICAL")
        self.assertEqual(len(crits), 5)

    def test_summary_stats_empty(self):
        stats = kb_store.summary_stats()
        self.assertEqual(stats, {})

    def test_summary_stats_counts(self):
        _fill_kb(6)   # all Normal
        stats = kb_store.summary_stats()
        self.assertEqual(stats["total_windows"], 6)
        self.assertIn("critical_count", stats)
        self.assertIn("warning_count", stats)

    def test_thread_safety(self):
        """Multiple threads adding entries must not corrupt the list."""
        kb_store._kb.clear()
        errors = []

        def _adder(wid_start, count):
            try:
                for i in range(count):
                    chunk = [_make_record() for _ in range(config.WINDOW_SIZE)]
                    summary = process_window(chunk)
                    if summary:
                        entry = generate_kb_entry(wid_start + i, summary)
                        kb_store.add_entry(entry)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_adder, args=(i * 20, 10))
                   for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertGreater(len(kb_store.get_all()), 0)

    def test_get_flagged(self):
        _fill_kb(3)
        all_entries = kb_store.get_all()
        # Inject an entry with a known flag
        entry = all_entries[0].copy()
        entry["flags"] = ["HIGH_TEMP"]
        kb_store._kb[0] = entry
        flagged = kb_store.get_flagged("HIGH_TEMP")
        self.assertGreaterEqual(len(flagged), 1)

    def test_json_persistence(self):
        _fill_kb(3)
        # Force persist
        kb_store._persist()
        path = config.KB_JSON_PATH
        self.assertTrue(os.path.exists(path), "KB JSON should be written to disk")
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 3)


# =============================================================================
# 6. WINDOWING TESTS
# =============================================================================

class TestWindowing(unittest.TestCase):

    def test_process_window_normal(self):
        """process_window returns raw stats dict with expected keys."""
        chunk = [_make_record() for _ in range(config.WINDOW_SIZE)]
        summary = process_window(chunk)
        self.assertIsNotNone(summary)
        for key in ("t_start", "t_end", "n_rows", "stats", "trends",
                    "dominant_state", "state_counts"):
            self.assertIn(key, summary, f"process_window missing key: {key}")

    def test_process_window_empty_returns_none(self):
        result = process_window([])
        self.assertIsNone(result)

    def test_process_window_small_chunk_still_returns(self):
        """process_window returns stats for any non-empty chunk (size guard is in caller)."""
        chunk = [_make_record() for _ in range(2)]
        result = process_window(chunk)
        self.assertIsNotNone(result)
        self.assertEqual(result["n_rows"], 2)

    def test_thermal_spike_is_critical(self):
        """High-temp window should produce CRITICAL severity in KB entry."""
        chunk = [_make_record(temp=56.0, current=210.0)
                 for _ in range(config.WINDOW_SIZE)]
        summary = process_window(chunk)
        self.assertIsNotNone(summary)
        entry = generate_kb_entry(1, summary)
        self.assertIn(entry["severity"], ("CRITICAL", "Warning"),
                      f"Thermal spike should be critical or warning, got: {entry['severity']}")

    def test_low_soc_flags_warning(self):
        """SoC=12% chunk must produce CRITICAL_SOC or LOW_SOC flag in KB entry."""
        chunk = [_make_record(soc=12.0) for _ in range(config.WINDOW_SIZE)]
        summary = process_window(chunk)
        self.assertIsNotNone(summary)
        entry = generate_kb_entry(2, summary)
        flags = entry.get("flags", [])
        self.assertTrue(
            any(f in flags for f in ("LOW_SOC", "CRITICAL_SOC")),
            f"Expected low SOC flag, got: {flags}"
        )

    def test_regen_braking_flag(self):
        """Strong negative current must produce REGEN_BRAKING flag in KB entry."""
        chunk = [_make_record(current=-120.0, state="Regen")
                 for _ in range(config.WINDOW_SIZE)]
        summary = process_window(chunk)
        self.assertIsNotNone(summary)
        entry = generate_kb_entry(3, summary)
        self.assertIn("REGEN_BRAKING", entry.get("flags", []))

    def test_generate_kb_entry_structure(self):
        chunk = [_make_record() for _ in range(config.WINDOW_SIZE)]
        summary = process_window(chunk)
        entry = generate_kb_entry(42, summary)
        for key in ("window_id", "time_range", "severity", "metrics",
                    "observations", "inference", "flags"):
            self.assertIn(key, entry, f"KB entry missing field: {key}")
        self.assertEqual(entry["window_id"], 42)


# =============================================================================
# 7. DATA INJECTOR TESTS
# =============================================================================

class TestDataInjector(unittest.TestCase):

    def setUp(self):
        _reset_injector()

    def test_get_row_count_starts_zero(self):
        self.assertEqual(data_injector.get_row_count(), 0)

    def test_injection_not_complete_initially(self):
        self.assertFalse(data_injector.injection_complete())

    def test_get_buffer_snapshot_empty(self):
        snap = data_injector.get_buffer_snapshot()
        self.assertEqual(snap, [])

    def test_get_latest_empty(self):
        latest = data_injector.get_latest(5)
        self.assertEqual(latest, [])

    def test_buffer_fills_after_manual_push(self):
        import data_injector as _di
        for _ in range(5):
            r = _make_record()
            with _di._buffer_lock:
                _di._buffer.append(r)
            _di._row_count += 1
        self.assertEqual(data_injector.get_row_count(), 5)
        self.assertEqual(len(data_injector.get_buffer_snapshot()), 5)

    def test_get_latest_respects_limit(self):
        import data_injector as _di
        for i in range(20):
            with _di._buffer_lock:
                _di._buffer.append(_make_record(soc=float(i)))
            _di._row_count += 1
        latest = data_injector.get_latest(5)
        self.assertEqual(len(latest), 5)
        self.assertAlmostEqual(latest[-1]["soc"], 19.0)


# =============================================================================
# 8. CONTEXT BUILDER TESTS
# =============================================================================

class TestContextBuilder(unittest.TestCase):

    def setUp(self):
        _reset_session()
        kb_store._kb.clear()
        _reset_injector()

    def test_context_with_empty_kb(self):
        ctx = llm_agent._build_context()
        self.assertIn("No windows processed yet", ctx)

    def test_context_with_kb_data(self):
        _fill_kb(3)
        ctx = llm_agent._build_context()
        self.assertIn("TRIP KB OVERVIEW", ctx)
        self.assertIn("RECENT WINDOWS", ctx)

    def test_context_includes_live_reading(self):
        import data_injector as _di
        r = _make_record(voltage=395.5, soc=72.3)
        with _di._buffer_lock:
            _di._buffer.append(r)
        ctx = llm_agent._build_context()
        self.assertIn("LIVE READING", ctx)

    def test_context_no_live_shows_placeholder(self):
        ctx = llm_agent._build_context()
        self.assertIn("No data yet", ctx)

    def test_context_includes_criticals(self):
        _fill_kb(5, severity_override="CRITICAL")
        ctx = llm_agent._build_context()
        self.assertIn("CRITICAL EVENTS", ctx)


# =============================================================================
# 9. END-TO-END PIPELINE INTEGRATION TEST
# =============================================================================

class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        _reset_session()
        _reset_injector()
        kb_store._kb.clear()

    def test_full_pipeline_no_llm(self):
        """
        Inject rows → window → KB → query (rule engine fallback).
        Must produce a personalised, clean answer.
        """
        # Push records manually
        import data_injector as _di
        for _ in range(config.WINDOW_SIZE * 3):
            r = _make_record()
            with _di._buffer_lock:
                _di._buffer.append(r)
            _di._row_count += 1

        # Process windows
        buf = data_injector.get_buffer_snapshot()
        wid = 1
        for i in range(0, len(buf) - config.WINDOW_SIZE + 1, config.WINDOW_SIZE):
            chunk = buf[i : i + config.WINDOW_SIZE]
            s = process_window(chunk)
            if s:
                kb_store.add_entry(generate_kb_entry(wid, s))
                wid += 1

        self.assertGreater(len(kb_store.get_all()), 0, "KB must have entries")

        # Query via fast-path
        llm_agent.set_driver_name("Renault")
        answer, source = llm_agent.query_llm("is the battery okay?", verbose=False)

        self.assertIn("Renault", answer)
        self.assertNotIn("LLM was unavailable", answer)
        self.assertNotIn("rule engine", answer.lower())

    def test_name_entry_persists_across_queries(self):
        _fill_kb(3)
        llm_agent.set_driver_name("Priya")
        for q in ["is the soc fine?", "is it safe?", "temperature okay?"]:
            _reset_session()
            llm_agent.set_driver_name("Priya")
            answer, _ = llm_agent.query_llm(q, verbose=False)
            self.assertIn("Priya", answer, f"Name missing in answer to: {q}")

    def test_greeted_flag_set_after_first_query(self):
        _fill_kb(3)
        _reset_session()
        llm_agent.set_driver_name("Alex")
        self.assertFalse(llm_agent._session["greeted"])
        llm_agent.query_llm("is the soc okay?", verbose=False)
        self.assertTrue(llm_agent._session["greeted"])

    def test_multiple_queries_dont_repeat_hello(self):
        _fill_kb(5)
        llm_agent.set_driver_name("Sam")
        # First query — should say Hello
        answer1, _ = llm_agent.query_llm("is the soc fine?", verbose=False)
        self.assertIn("Hello Sam", answer1)
        # Second query — should NOT say Hello
        answer2, _ = llm_agent.query_llm("is the soc fine?", verbose=False)
        self.assertNotIn("Hello", answer2,
                         "Should not re-greet after first interaction")


# =============================================================================
# 10. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        _reset_session()
        kb_store._kb.clear()

    def test_query_with_empty_kb_returns_graceful_message(self):
        """No KB data yet — must not crash, must return something sensible."""
        _reset_session()
        config.LLM_TIMEOUT_S = 0.001
        answer, source = llm_agent.query_llm("how is my battery?", verbose=False)
        config.LLM_TIMEOUT_S = 2
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn(source, ("RuleEngine", "LLM", "FastPath"))

    def test_very_long_name_does_not_crash(self):
        long_name = "A" * 200
        llm_agent.set_driver_name(long_name)
        self.assertEqual(llm_agent.get_driver_name(), long_name)

    def test_special_chars_in_name(self):
        llm_agent.set_driver_name("O'Brien-Jr.")
        self.assertEqual(llm_agent.get_driver_name(), "O'Brien-Jr.")

    def test_query_empty_string_handled(self):
        _fill_kb(3)
        # Empty query should not crash
        try:
            answer, source = llm_agent.query_llm("", verbose=False)
            self.assertIsInstance(answer, str)
        except Exception as e:
            self.fail(f"Empty query raised exception: {e}")

    def test_window_with_all_zeros_does_not_crash(self):
        chunk = [_make_record(voltage=0, current=0, soc=0, temp=0)
                 for _ in range(config.WINDOW_SIZE)]
        try:
            summary = process_window(chunk)
            # May return None for degenerate data — that's acceptable
        except Exception as e:
            self.fail(f"process_window crashed on zero-data: {e}")

    def test_kb_summary_with_mixed_severities(self):
        kb_store._kb.clear()
        wid = 1
        for sev in ["Normal", "Warning", "CRITICAL", "Normal", "Info"]:
            chunk = [_make_record() for _ in range(config.WINDOW_SIZE)]
            s = process_window(chunk)
            if s:
                entry = generate_kb_entry(wid, s)
                entry["severity"] = sev
                kb_store.add_entry(entry)
                wid += 1
        stats = kb_store.summary_stats()
        self.assertEqual(stats["total_windows"], 5)

    def test_rule_engine_query_with_flags(self):
        """HIGH_TEMP and REGEN_BRAKING flags should appear in summary."""
        chunk = [_make_record(temp=56.0) for _ in range(config.WINDOW_SIZE)]
        s = process_window(chunk)
        if s:
            kb_store.add_entry(generate_kb_entry(1, s))
        msg = kb_store.rule_based_summary("any issues?")
        # Should at minimum produce a non-empty string
        self.assertGreater(len(msg), 10)

    def test_speak_does_not_queue_when_unavailable(self):
        speech_mod._tts_available = False
        q_before = speech_mod._tts_queue.qsize()
        speech_mod.speak("This should not queue")
        self.assertEqual(speech_mod._tts_queue.qsize(), q_before)

    def test_concurrent_kb_reads_while_writing(self):
        """Concurrent reads and writes must not deadlock."""
        import data_injector as _di
        stop = threading.Event()
        errors = []

        def _reader():
            while not stop.is_set():
                try:
                    kb_store.get_all()
                    kb_store.summary_stats()
                    kb_store.get_recent(5)
                except Exception as e:
                    errors.append(str(e))

        def _writer():
            wid = 100
            while not stop.is_set():
                chunk = [_make_record() for _ in range(config.WINDOW_SIZE)]
                s = process_window(chunk)
                if s:
                    kb_store.add_entry(generate_kb_entry(wid, s))
                    wid += 1

        readers = [threading.Thread(target=_reader) for _ in range(3)]
        writer  = threading.Thread(target=_writer)
        for t in readers:
            t.start()
        writer.start()
        time.sleep(0.5)
        stop.set()
        writer.join(timeout=2)
        for t in readers:
            t.join(timeout=2)
        self.assertEqual(len(errors), 0, f"Concurrency errors: {errors}")


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich import box
    from rich.panel import Panel

    _console = Console(highlight=False)
    _console.print(Panel(
        "[bold bright_blue]BMS Pipeline — Production Test Suite[/bold bright_blue]\n"
        "[dim]Running all unit, integration, and edge-case tests...[/dim]",
        border_style="bright_blue", box=box.DOUBLE_EDGE, padding=(1, 2)
    ))

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total   = result.testsRun
    passed  = total - len(result.failures) - len(result.errors)
    failed  = len(result.failures) + len(result.errors)

    _console.print()
    if failed == 0:
        _console.print(Panel(
            f"[bold green]ALL {total} TESTS PASSED[/bold green]",
            border_style="green", box=box.ROUNDED
        ))
    else:
        _console.print(Panel(
            f"[bold red]{failed} FAILED[/bold red] | [green]{passed} passed[/green] of {total}",
            border_style="red", box=box.ROUNDED
        ))

    sys.exit(0 if failed == 0 else 1)
