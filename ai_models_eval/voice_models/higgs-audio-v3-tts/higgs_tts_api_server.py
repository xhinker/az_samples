#!/usr/bin/env python3
"""Higgs Audio v3 TTS — OpenAI-compatible API server.

Async aiohttp server with batch and streaming TTS endpoints.
Handles long text automatically via chunked synthesis.

Usage:
    python3 higgs_tts_api_server.py --device cuda:0 --port 8081
"""

import argparse
import asyncio
import base64
import hashlib
from functools import lru_cache
import io
import json
import logging
import os
import sys
import tempfile
import wave
from pathlib import Path

import aiohttp
from aiohttp import web
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from higgs_audio_infer import HiggsTTS, VOICE_PRESETS, SAMPLE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

tts_engine = None
generation_lock = asyncio.Lock()


# --- Endpoints ---

async def health_check(request):
    return web.json_response({
        "status": "ok",
        "model_loaded": tts_engine is not None,
        "model_codebooks": tts_engine.num_codebooks if tts_engine else None,
        "sample_rate": SAMPLE_RATE,
        "device": str(tts_engine.model.device) if tts_engine else None,
    })


async def list_models(request):
    return web.json_response({
        "object": "list",
        "data": [{
            "id": "higgs-audio-v3-tts-4b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "bosonai",
        }],
    })


# --- Reference audio cache ---
# Cache encoded delayed_ref tensors to avoid re-encoding on every request.
# Key: SHA-256 hash of the base64 input (deterministic, fixed-length).
# Max 16 entries — enough for typical usage, bounded VRAM/CPU memory.
_ref_cache = {}
_REF_CACHE_MAX = 16
_ref_cache_order = []  # LRU tracking


def _encode_reference_audio_raw(b64_string):
    """Decode base64 reference audio and encode via model. Returns delayed_ref tensor (CPU)."""
    ref_bytes = base64.b64decode(b64_string)
    try:
        with wave.open(io.BytesIO(ref_bytes), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        ref_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        ref = tts_engine.encode_reference_audio(ref_np, sr)
    except Exception:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(ref_bytes)
        tmp.close()
        ref = tts_engine.encode_reference_audio(tmp.name)
        os.unlink(tmp.name)
    # Store on CPU to save VRAM; will be moved to GPU when used
    return ref.cpu()


def _get_reference_audio(b64_string):
    """Get delayed_ref from cache or encode (with LRU eviction)."""
    key = hashlib.sha256(b64_string.encode()).hexdigest()

    if key in _ref_cache:
        # Move to front (LRU)
        _ref_cache_order.remove(key)
        _ref_cache_order.append(key)
        logger.info("Reference audio cache HIT: %s", key[:8])
        return _ref_cache[key]

    logger.info("Reference audio cache MISS: %s, encoding...", key[:8])
    ref = _encode_reference_audio_raw(b64_string)

    # Evict oldest if at capacity
    if len(_ref_cache) >= _REF_CACHE_MAX:
        oldest = _ref_cache_order.pop(0)
        evicted = _ref_cache.pop(oldest)
        del evicted  # Free memory
        logger.debug("Cache evicted: %s (size=%d)", oldest[:8], len(_ref_cache))

    _ref_cache[key] = ref
    _ref_cache_order.append(key)
    return ref


def _build_tts_params(body):
    """Extract common TTS parameters from request body."""
    return {
        "text_input": body.get("input", "").strip(),
        "voice": body.get("voice", "alloy"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "reference_audio": None,
        "reference_text": body.get("reference_text"),
    }


async def audio_speech(request):
    """POST /v1/audio/speech — Batch TTS (returns full audio)."""
    body = await request.json()
    params = _build_tts_params(body)

    if not params["text_input"]:
        return web.json_response({"error": "Missing 'input' text"}, status=400)

    if body.get("reference_audio"):
        params["reference_audio"] = _get_reference_audio(body["reference_audio"])

    fmt = body.get("response_format", "wav")

    async with generation_lock:
        try:
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            out_path, duration = await loop.run_in_executor(
                None, lambda: tts_engine.synthesize(**{
                    k: v for k, v in params.items() if k != "text_input"
                }, text_input=params["text_input"])
            )

            audio_bytes = Path(out_path).read_bytes()
            os.unlink(out_path)

            if fmt == "mp3":
                mp3_bytes = await _convert_to_mp3(audio_bytes)
                return web.Response(body=mp3_bytes, content_type="audio/mpeg")
            elif fmt == "pcm":
                # Strip WAV header (44 bytes) for raw PCM
                pcm_bytes = audio_bytes[44:]
                return web.Response(body=pcm_bytes, content_type="audio/pcm",
                                    headers={"X-Sample-Rate": str(SAMPLE_RATE)})
            else:
                return web.Response(body=audio_bytes, content_type="audio/wav")

        except Exception as e:
            logger.error("Batch synthesis failed: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)


async def audio_speech_stream(request):
    """POST /v1/audio/speech-stream — Streaming TTS (SSE + PCM chunks)."""
    body = await request.json()
    params = _build_tts_params(body)

    if not params["text_input"]:
        return web.json_response({"error": "Missing 'input' text"}, status=400)

    if body.get("reference_audio"):
        params["reference_audio"] = _get_reference_audio(body["reference_audio"])

    async with generation_lock:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Sample-Rate": str(SAMPLE_RATE),
            }
        )
        await response.prepare(request)

        # Send metadata
        meta = {"type": "metadata", "sample_rate": SAMPLE_RATE,
                "channels": 1, "format": "pcm16le", "voice": params["voice"]}
        await response.write(f"event: metadata\ndata: {json.dumps(meta)}\n\n".encode())

        total_bytes = 0
        chunk_count = 0

        try:
            # stream() handles long text splitting + incremental yielding
            for pcm_chunk in tts_engine.stream(
                text_input=params["text_input"],
                voice=params["voice"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                reference_audio=params["reference_audio"],
                reference_text=params["reference_text"],
                chunk_size=960,
                decode_every=50,
            ):
                data_b64 = base64.b64encode(pcm_chunk).decode("ascii")
                event = {"type": "audio", "index": chunk_count,
                         "bytes": len(pcm_chunk), "data": data_b64}
                await response.write(f"event: audio\ndata: {json.dumps(event)}\n\n".encode())
                total_bytes += len(pcm_chunk)
                chunk_count += 1

        except Exception as e:
            logger.error("Streaming failed: %s", e, exc_info=True)
            error_event = {"type": "error", "message": str(e)}
            await response.write(f"event: error\ndata: {json.dumps(error_event)}\n\n".encode())

        # Send done
        done = {"type": "done", "total_bytes": total_bytes,
                "total_chunks": chunk_count,
                "duration_seconds": round(total_bytes / 2 / SAMPLE_RATE, 2)}
        await response.write(f"event: done\ndata: {json.dumps(done)}\n\n".encode())

        return response


async def _convert_to_mp3(wav_bytes):
    """Convert WAV bytes to MP3 using ffmpeg."""
    try:
        import subprocess
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-codec:a", "libmp3lame", "-q:a", "2", "pipe:1"],
            input=wav_bytes, capture_output=True, timeout=30,
        )
        if proc.returncode == 0:
            return proc.stdout
        logger.warning("ffmpeg MP3 conversion failed: %s", proc.stderr.decode())
    except FileNotFoundError:
        logger.warning("ffmpeg not found, returning WAV as-is")
    except Exception as e:
        logger.warning("MP3 conversion error: %s", e)
    return wav_bytes  # Fall back to WAV


async def index_page(request):
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return web.FileResponse(static_path)
    return web.Response(text="Web UI not found.", status=404)


def create_app():
    app = web.Application()
    app.router.add_get("/", index_page)
    app.router.add_get("/v1/health", health_check)
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/audio/speech", audio_speech)
    app.router.add_post("/v1/audio/speech-stream", audio_speech_stream)

    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.router.add_static("/static/", path=str(static_path), name="static")
    return app


async def main():
    global tts_engine

    parser = argparse.ArgumentParser(description="Higgs Audio v3 TTS API Server")
    parser.add_argument("--host", default=os.environ.get("HIGGS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HIGGS_PORT", "8081")))
    parser.add_argument("--model-path", default=os.environ.get(
        "HIGGS_MODEL_PATH", "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"))
    parser.add_argument("--device", default=os.environ.get("HIGGS_DEVICE", "cuda:0"))
    parser.add_argument("--dtype", default=os.environ.get("HIGGS_DTYPE", "bfloat16"),
                        choices=["bfloat16", "float16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"\n{'='*60}")
    print(f"  Higgs Audio v3 TTS — API Server")
    print(f"{'='*60}")
    print(f"  Model:  {args.model_path}")
    print(f"  Device: {args.device}  |  Dtype: {args.dtype}")
    print(f"  URL:    http://localhost:{args.port}")
    print(f"{'='*60}\n")

    tts_engine = HiggsTTS(model_path=args.model_path, device=args.device, dtype=dtype)
    tts_engine.get_vram_info()

    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    logger.info("Server ready at http://localhost:%d", args.port)
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
