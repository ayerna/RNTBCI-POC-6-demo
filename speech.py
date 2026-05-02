"""
speech.py - Speech-to-Text (Whisper) + Text-to-Speech (pyttsx3)
Gracefully degrades if hardware or libraries are unavailable.
"""

import io
import threading
import logger
from config import (WHISPER_MODEL, MIC_RECORD_SECS,
                    TTS_RATE, TTS_VOLUME, SPEECH_ENABLED)

# ── Lazy imports with availability flags ───────────────────────────────────────
_whisper_model   = None
_tts_engine      = None
_stt_available   = False
_tts_available   = False


def _init_tts() -> bool:
    global _tts_engine, _tts_available
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate",   TTS_RATE)
        _tts_engine.setProperty("volume", TTS_VOLUME)
        # prefer a female voice if available
        voices = _tts_engine.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower():
                _tts_engine.setProperty("voice", v.id)
                break
        _tts_available = True
        logger.log_speech("TTS engine (pyttsx3) ready.")
        return True
    except Exception as e:
        logger.log_speech(f"TTS unavailable: {e}")
        return False


def _init_stt() -> bool:
    global _whisper_model, _stt_available
    try:
        import whisper as _w
        logger.log_speech(f"Loading Whisper model '{WHISPER_MODEL}' (first run downloads weights)...")
        _whisper_model = _w.load_model(WHISPER_MODEL)
        _stt_available = True
        logger.log_speech(f"Whisper STT model '{WHISPER_MODEL}' loaded.")
        return True
    except Exception as e:
        logger.log_speech(f"STT unavailable: {e}")
        return False


def init_speech() -> tuple[bool, bool]:
    """
    Initialise TTS and STT engines.
    Returns (tts_ok, stt_ok).
    Call once at startup.
    """
    if not SPEECH_ENABLED:
        logger.log_speech("Speech disabled via config (SPEECH_ENABLED=False).")
        return False, False
    tts_ok = _init_tts()
    stt_ok = _init_stt()
    return tts_ok, stt_ok


# ── Text-to-Speech ─────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    """
    Speak the given text aloud via pyttsx3.
    Blocks until speech is complete. Falls back to silent print if unavailable.
    """
    if not _tts_available or not SPEECH_ENABLED:
        return   # silent fallback — caller already prints the text

    # pyttsx3 must run from main thread on Windows; use a lock if multi-threaded
    try:
        # Truncate very long responses to avoid overly long speech
        truncated = text if len(text) <= 600 else text[:600] + "... (response truncated for speech)"
        _tts_engine.say(truncated)
        _tts_engine.runAndWait()
    except RuntimeError:
        # Already running — skip
        pass
    except Exception as e:
        logger.log_speech(f"TTS error: {e}")


# ── Speech-to-Text ─────────────────────────────────────────────────────────────

def listen() -> str | None:
    """
    Capture MIC_RECORD_SECS of audio from the default microphone,
    transcribe with Whisper, and return the text.
    Returns None if STT is unavailable or no speech detected.
    """
    if not _stt_available or not SPEECH_ENABLED:
        logger.log_speech("STT not available. Please type your query instead.")
        return None

    try:
        import sounddevice as sd
        import numpy as np
        import whisper as _w

        logger.log_speech(f"Listening... speak now ({MIC_RECORD_SECS}s window)")
        sample_rate = 16_000
        audio = sd.rec(
            int(MIC_RECORD_SECS * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()   # block until recording done
        audio_flat = audio.flatten()

        logger.log_speech("Transcribing audio with Whisper...")
        result = _whisper_model.transcribe(audio_flat, language="en", fp16=False)
        text   = result.get("text", "").strip()

        if not text:
            logger.log_speech("No speech detected.")
            return None

        logger.log_speech(f'Transcribed: "{text}"')
        return text

    except Exception as e:
        logger.log_speech(f"STT error during recording: {e}")
        return None
