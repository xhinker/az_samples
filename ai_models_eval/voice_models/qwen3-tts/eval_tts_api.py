#!/usr/bin/env python3
"""Evaluation script for the OpenAI-compatible Qwen3-TTS API.

Tests the TTS service running at http://127.0.0.1:8090/v1/audio/speech

Usage:
  # Make sure tts_api_server.py is running first:
  #   python3 tts_api_server.py \
  #     --voice-clone-model-id models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main

  python3 eval_tts_api.py                          # run all tests
  python3 eval_tts_api.py --text "Hello"            # quick single test
  python3 eval_tts_api.py --format wav --output out.wav

  # Voice clone mode (Base model):
  python3 eval_tts_api.py \
    --text "今夜月色真好。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "reference transcript" \
    --output clone.wav --play
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is not installed. Run: pip install requests")
    sys.exit(1)


BASE_URL = "http://127.0.0.1:8090"
DEFAULT_VOICE = "vivian"
DEFAULT_FORMAT = "wav"


@dataclass
class TestResult:
    name: str
    passed: bool
    latency_first_chunk_ms: Optional[float] = None
    latency_total_ms: Optional[float] = None
    audio_bytes: int = 0
    error: Optional[str] = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _health_check(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"  [health] status={data.get('status')}  models_loaded={data.get('models_loaded')}")
        return True
    except Exception as exc:
        print(f"  [health] FAILED: {exc}")
        return False


def _call_tts(
    base_url: str,
    text: str,
    voice: str = DEFAULT_VOICE,
    response_format: str = DEFAULT_FORMAT,
    speed: float = 1.0,
    timeout: int = 120,
    ref_audio_path: Optional[Path] = None,
    ref_text: Optional[str] = None,
) -> tuple[bytes, float, float]:
    """Call the TTS API and return (audio_bytes, first_chunk_latency_ms, total_latency_ms).

    When ref_audio_path + ref_text are provided, voice_clone mode is used.
    """
    payload: dict = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }
    if ref_audio_path and ref_text:
        payload["reference_audio"] = base64.b64encode(Path(ref_audio_path).read_bytes()).decode()
        payload["reference_text"] = ref_text
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/audio/speech",
        json=payload,
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    buf = io.BytesIO()
    first_chunk_ms: Optional[float] = None
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - t0) * 1000
            buf.write(chunk)

    total_ms = (time.perf_counter() - t0) * 1000
    audio_bytes = buf.getvalue()
    return audio_bytes, first_chunk_ms or total_ms, total_ms


def _play_audio(audio_bytes: bytes, fmt: str, sample_rate: int = 24000) -> None:
    """Try to play audio via aplay (wav/pcm) or ffplay (any)."""
    if fmt == "pcm":
        # wrap in WAV for playback
        import struct
        sr = sample_rate
        data_size = len(audio_bytes)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", data_size + 36, b"WAVE",
            b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
            b"data", data_size,
        )
        audio_bytes = header + audio_bytes
        fmt = "wav"

    if fmt == "wav":
        try:
            proc = subprocess.run(
                ["aplay", "-q", "-"],
                input=audio_bytes,
                timeout=60,
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Fallback: ffplay
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-i", "pipe:0"],
            input=audio_bytes,
            timeout=60,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  [play] No audio player found (tried aplay, ffplay). Save with --output.")


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------

def test_health(base_url: str) -> TestResult:
    print("\n[Test] health check")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        ok = data.get("status") == "ok"
        print(f"  Response: {json.dumps(data, indent=2)}")
        return TestResult("health_check", passed=ok, details=data)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("health_check", passed=False, error=str(exc))


def test_models_list(base_url: str) -> TestResult:
    print("\n[Test] GET /v1/models")
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        ids = [m["id"] for m in data.get("data", [])]
        print(f"  Models: {ids}")
        return TestResult("models_list", passed=bool(ids), details=data)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("models_list", passed=False, error=str(exc))


def test_speech_wav(base_url: str) -> TestResult:
    text = "Hello! This is a test of the Qwen3 TTS API service."
    print(f"\n[Test] WAV speech  |  text={text!r}")
    try:
        audio, fc_ms, total_ms = _call_tts(base_url, text, response_format="wav")
        ok = len(audio) > 44  # at least WAV header + some audio
        print(f"  First chunk: {fc_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  Audio size: {len(audio)} bytes")
        return TestResult("speech_wav", passed=ok, latency_first_chunk_ms=fc_ms, latency_total_ms=total_ms, audio_bytes=len(audio))
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("speech_wav", passed=False, error=str(exc))


def test_speech_pcm(base_url: str) -> TestResult:
    text = "Streaming raw PCM audio output."
    print(f"\n[Test] PCM speech  |  text={text!r}")
    try:
        audio, fc_ms, total_ms = _call_tts(base_url, text, response_format="pcm")
        ok = len(audio) > 0 and len(audio) % 2 == 0  # PCM16 is 2 bytes per sample
        print(f"  First chunk: {fc_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  Samples: {len(audio)//2}")
        return TestResult("speech_pcm", passed=ok, latency_first_chunk_ms=fc_ms, latency_total_ms=total_ms, audio_bytes=len(audio))
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("speech_pcm", passed=False, error=str(exc))


def test_openai_voice_names(base_url: str) -> TestResult:
    """Verify OpenAI voice names (alloy, echo, ...) are accepted."""
    text = "Testing OpenAI voice name compatibility."
    print(f"\n[Test] OpenAI voice names  |  text={text!r}")
    voices = ["alloy", "echo", "nova", "shimmer"]
    passed_voices = []
    failed_voices = []
    for voice in voices:
        try:
            audio, fc_ms, _ = _call_tts(base_url, text, voice=voice, response_format="pcm")
            if len(audio) > 0:
                passed_voices.append(voice)
                print(f"  {voice}: OK ({len(audio)} bytes, first_chunk={fc_ms:.0f}ms)")
            else:
                failed_voices.append(voice)
                print(f"  {voice}: EMPTY audio")
        except Exception as exc:
            failed_voices.append(voice)
            print(f"  {voice}: FAILED ({exc})")
    ok = len(failed_voices) == 0
    return TestResult(
        "openai_voice_names",
        passed=ok,
        details={"passed": passed_voices, "failed": failed_voices},
    )


def test_long_text(base_url: str) -> TestResult:
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "In the beginning, the universe was created. "
        "This has made a lot of people very angry and has been widely regarded as a bad move. "
        "Many solutions were proposed, but most of these dealt with the question of whether "
        "to use tabs or spaces, which was widely seen as a bit of a non-starter. "
        "It is a truth universally acknowledged that a single person in possession of "
        "a good fortune must be in want of a great AI voice assistant."
    )
    print(f"\n[Test] Long text ({len(text)} chars)")
    try:
        audio, fc_ms, total_ms = _call_tts(base_url, text, response_format="wav", timeout=300)
        ok = len(audio) > 1000
        duration_s = len(audio) / (24000 * 2) if ok else 0
        print(
            f"  First chunk: {fc_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  "
            f"Audio size: {len(audio)} bytes (~{duration_s:.1f}s)"
        )
        return TestResult("long_text", passed=ok, latency_first_chunk_ms=fc_ms, latency_total_ms=total_ms, audio_bytes=len(audio))
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("long_text", passed=False, error=str(exc))


def test_empty_input(base_url: str) -> TestResult:
    print("\n[Test] Empty input (should return 400)")
    try:
        r = requests.post(
            f"{base_url}/v1/audio/speech",
            json={"model": "tts-1", "input": "", "voice": "alloy"},
            timeout=15,
        )
        ok = r.status_code == 400
        print(f"  Status: {r.status_code}  (expected 400)")
        return TestResult("empty_input", passed=ok, details={"status_code": r.status_code})
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("empty_input", passed=False, error=str(exc))


def test_voice_clone(base_url: str, ref_audio_path: Path, ref_text: str) -> TestResult:
    text = "This is a voice clone test. The synthesized voice should match the reference audio."
    print(f"\n[Test] Voice clone  |  ref={ref_audio_path.name}  |  text={text!r}")
    try:
        audio, fc_ms, total_ms = _call_tts(
            base_url, text, response_format="wav",
            ref_audio_path=ref_audio_path, ref_text=ref_text,
            timeout=300,
        )
        ok = len(audio) > 44
        duration_s = (len(audio) - 44) / (24000 * 2) if ok else 0
        print(f"  First chunk: {fc_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  Audio: {len(audio)} bytes (~{duration_s:.1f}s)")
        return TestResult("voice_clone", passed=ok, latency_first_chunk_ms=fc_ms, latency_total_ms=total_ms, audio_bytes=len(audio))
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("voice_clone", passed=False, error=str(exc))


def test_invalid_format(base_url: str) -> TestResult:
    print("\n[Test] Invalid response_format (should return 400)")
    try:
        r = requests.post(
            f"{base_url}/v1/audio/speech",
            json={"model": "tts-1", "input": "test", "voice": "alloy", "response_format": "ogg"},
            timeout=15,
        )
        ok = r.status_code == 400
        print(f"  Status: {r.status_code}  (expected 400)")
        return TestResult("invalid_format", passed=ok, details={"status_code": r.status_code})
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return TestResult("invalid_format", passed=False, error=str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(results: list[TestResult]) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        extra = ""
        if r.latency_first_chunk_ms is not None:
            extra += f"  first_chunk={r.latency_first_chunk_ms:.0f}ms"
        if r.latency_total_ms is not None:
            extra += f"  total={r.latency_total_ms:.0f}ms"
        if r.error:
            extra += f"  error={r.error!r}"
        print(f"  [{status}] {r.name}{extra}")
    print(f"\n{passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Qwen3-TTS OpenAI-compatible API")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of TTS API server")
    parser.add_argument("--text", default=None, help="Single text to synthesize (skips full test suite)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name")
    parser.add_argument("--format", dest="fmt", default=DEFAULT_FORMAT, choices=["wav", "pcm", "mp3"], help="Response format")
    parser.add_argument("--output", default=None, help="Save audio to file (e.g. out.wav)")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")
    parser.add_argument("--quick", action="store_true", help="Run only fast tests (skip long text and voice clone)")
    parser.add_argument("--ref-audio", default=None, help="Path to reference WAV for voice clone mode")
    parser.add_argument(
        "--ref-text",
        default="今夜的月光如此清亮，不做些什么真是浪费。随我一同去月下漫步吧，不许拒绝。",
        help="Transcript of the reference audio",
    )
    args = parser.parse_args()

    ref_audio_path: Optional[Path] = Path(args.ref_audio) if args.ref_audio else None

    print(f"Target: {args.url}")

    if args.text:
        # Single synthesis mode
        print(f"\nSynthesizing: {args.text!r}")
        if ref_audio_path:
            print(f"Mode: voice_clone  |  ref={ref_audio_path}  |  ref_text={args.ref_text!r}")
        else:
            print(f"Mode: custom_voice  |  Voice: {args.voice}  |  Format: {args.fmt}")
        try:
            audio, fc_ms, total_ms = _call_tts(
                args.url, args.text, voice=args.voice, response_format=args.fmt,
                ref_audio_path=ref_audio_path, ref_text=args.ref_text if ref_audio_path else None,
            )
            print(f"First chunk: {fc_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  Audio: {len(audio)} bytes")
            if args.output:
                Path(args.output).write_bytes(audio)
                print(f"Saved to: {args.output}")
            if args.play:
                _play_audio(audio, args.fmt)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        return

    # Full evaluation suite
    print("\nRunning evaluation suite...")
    results: list[TestResult] = []

    results.append(test_health(args.url))
    results.append(test_models_list(args.url))
    results.append(test_empty_input(args.url))
    results.append(test_invalid_format(args.url))

    # These tests require the model to be loaded (slower)
    if _health_check(args.url):
        results.append(test_speech_wav(args.url))
        results.append(test_speech_pcm(args.url))
        results.append(test_openai_voice_names(args.url))
        if not args.quick:
            results.append(test_long_text(args.url))
            if ref_audio_path and ref_audio_path.exists():
                results.append(test_voice_clone(args.url, ref_audio_path, args.ref_text))
            else:
                print("\n[Skip] voice_clone test — pass --ref-audio <path> to enable it")

    print_summary(results)


if __name__ == "__main__":
    main()
