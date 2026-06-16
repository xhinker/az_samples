#!/usr/bin/env python3
"""Higgs Audio v3 TTS — OpenAI-compatible API server.

Built with aiohttp for async streaming. Supports batch TTS, streaming TTS,
voice cloning, and emotion/style control tokens.

Usage:
    python3 higgs_tts_api_server.py \
        --model-path /mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b \
        --device cuda:0 --port 8081

Endpoints:
    GET  /v1/health              — Health check
    GET  /v1/models              — List models (OpenAI-compatible)
    POST /v1/audio/speech        — Batch TTS (returns full audio)
    POST /v1/audio/speech-stream — Streaming TTS (SSE + PCM chunks)
    GET  /                       — Web test page
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import aiohttp
from aiohttp import web
import numpy as np
import torch

# Add project root to path for higgs_audio_infer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from higgs_audio_infer import HiggsTTS, VOICE_PRESETS, SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Globals ---
tts_engine = None
generation_lock = asyncio.Lock()  # One generation at a time


async def health_check(request):
    """GET /v1/health — Model status and device info."""
    return web.json_response({
        "status": "ok",
        "model_loaded": tts_engine is not None,
        "model_codebooks": tts_engine.num_codebooks if tts_engine else None,
        "sample_rate": SAMPLE_RATE,
        "device": str(tts_engine.model.device) if tts_engine else None,
    })


async def list_models(request):
    """GET /v1/models — OpenAI-compatible model list."""
    return web.json_response({
        "object": "list",
        "data": [{
            "id": "higgs-audio-v3-tts-4b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "bosonai",
        }],
    })


async def audio_speech(request):
    """POST /v1/audio/speech — Batch TTS (returns full audio file).

    Request body:
        input (str, required): Text to synthesize.
        voice (str, optional): Voice preset — alloy/echo/fable/onyx/nova/shimmer.
        response_format (str, optional): wav|pcm|mp3. Default: wav.
        temperature (float, optional): Override voice preset temperature.
        top_p (float, optional): Nucleus sampling threshold.
        top_k (int, optional): Top-k sampling limit.
        reference_audio (str, optional): Base64-encoded WAV for voice cloning.
        reference_text (str, optional): Transcript of reference audio.
    """
    body = await request.json()
    text_input = body.get("input", "")
    if not text_input.strip():
        return web.json_response({"error": "Missing 'input' text"}, status=400)

    voice = body.get("voice", "alloy")
    response_format = body.get("response_format", "wav")
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    top_k = body.get("top_k")
    reference_audio_b64 = body.get("reference_audio")
    reference_text = body.get("reference_text")

    async with generation_lock:
        # Decode reference audio if provided
        delayed_ref = None
        if reference_audio_b64:
            ref_bytes = base64.b64decode(reference_audio_b64)
            ref_path = None
            # Try to open as WAV directly from bytes
            try:
                wav_file = wave.open(io.BytesIO(ref_bytes), "rb")
                sr = wav_file.getframerate()
                nframes = wav_file.getnframes()
                raw = wav_file.readframes(nframes)
                wav_file.close()
                ref_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                delayed_ref = tts_engine.encode_reference_audio(ref_np, sr)
            except Exception:
                # Fall back to temp file
                ref_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                ref_path.write(ref_bytes)
                ref_path.close()
                delayed_ref = tts_engine.encode_reference_audio(ref_path.name)

        # Synthesize
        if response_format == "mp3":
            # Generate WAV first, then convert
            out_path, duration = tts_engine.synthesize(
                text_input=text_input,
                voice=voice, temperature=temperature, top_k=top_k, top_p=top_p,
                reference_audio=delayed_ref, reference_text=reference_text,
            )
            # Convert to MP3 via ffmpeg
            mp3_path = out_path.replace(".wav", ".mp3")
            os.system(f"ffmpeg -y -i {out_path} -codec:a libmp3lame -q:a 2 {mp3_path} 2>/dev/null")
            if os.path.exists(mp3_path):
                audio_bytes = Path(mp3_path).read_bytes()
                os.unlink(mp3_path)
                os.unlink(out_path)
            else:
                audio_bytes = Path(out_path).read_bytes()
                os.unlink(out_path)
            return web.Response(body=audio_bytes, content_type="audio/mpeg")
        elif response_format == "pcm":
            all_pcm = b""
            for chunk in tts_engine.stream(
                text_input=text_input,
                voice=voice, temperature=temperature, top_k=top_k, top_p=top_p,
                reference_audio=delayed_ref, reference_text=reference_text,
                chunk_size=4096,
            ):
                all_pcm += chunk
            return web.Response(body=all_pcm, content_type="audio/pcm",
                                headers={"X-Sample-Rate": str(SAMPLE_RATE)})
        else:
            # WAV (default)
            out_path, duration = tts_engine.synthesize(
                text_input=text_input,
                voice=voice, temperature=temperature, top_k=top_k, top_p=top_p,
                reference_audio=delayed_ref, reference_text=reference_text,
            )
            audio_bytes = Path(out_path).read_bytes()
            os.unlink(out_path)
            return web.Response(body=audio_bytes, content_type="audio/wav")


async def audio_speech_stream(request):
    """POST /v1/audio/speech-stream — Streaming TTS (SSE + PCM chunks).

    Returns Server-Sent Events with PCM16LE audio chunks.
    First event is metadata, subsequent events are audio data.

    Request body: same as /v1/audio/speech.
    """
    body = await request.json()
    text_input = body.get("input", "")
    if not text_input.strip():
        return web.json_response({"error": "Missing 'input' text"}, status=400)

    voice = body.get("voice", "alloy")
    response_format = body.get("response_format", "pcm")
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    top_k = body.get("top_k")
    reference_audio_b64 = body.get("reference_audio")
    reference_text = body.get("reference_text")

    async with generation_lock:
        # Decode reference audio if provided
        delayed_ref = None
        if reference_audio_b64:
            ref_bytes = base64.b64decode(reference_audio_b64)
            try:
                wav_file = wave.open(io.BytesIO(ref_bytes), "rb")
                sr = wav_file.getframerate()
                nframes = wav_file.getnframes()
                raw = wav_file.readframes(nframes)
                wav_file.close()
                ref_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                delayed_ref = tts_engine.encode_reference_audio(ref_np, sr)
            except Exception:
                ref_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                ref_path.write(ref_bytes)
                ref_path.close()
                delayed_ref = tts_engine.encode_reference_audio(ref_path.name)

        response_format = response_format or "pcm"

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

        # Send metadata event
        meta = {
            "type": "metadata",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "format": "pcm16le",
            "voice": voice,
        }
        await response.write(f"event: metadata\ndata: {json.dumps(meta)}\n\n".encode())

        # Send audio chunks as SSE
        total_bytes = 0
        chunk_count = 0
        try:
            for pcm_chunk in tts_engine.stream(
                text_input=text_input,
                voice=voice, temperature=temperature, top_k=top_k, top_p=top_p,
                reference_audio=delayed_ref, reference_text=reference_text,
                chunk_size=960,
            ):
                chunk_data = base64.b64encode(pcm_chunk).decode("ascii")
                event = {
                    "type": "audio",
                    "index": chunk_count,
                    "bytes": len(pcm_chunk),
                    "data": chunk_data,
                }
                await response.write(f"event: audio\ndata: {json.dumps(event)}\n\n".encode())
                total_bytes += len(pcm_chunk)
                chunk_count += 1
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)

        # Send done event
        done = {
            "type": "done",
            "total_bytes": total_bytes,
            "total_chunks": chunk_count,
            "duration_seconds": round(total_bytes / 2 / SAMPLE_RATE, 2),
        }
        await response.write(f"event: done\ndata: {json.dumps(done)}\n\n".encode())

        return response


async def index_page(request):
    """GET / — Web test page."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return web.FileResponse(static_path)
    return web.Response(text="Web UI not found. Place static/index.html in project root.", status=404)


def create_app(args):
    app = web.Application()
    app.router.add_get("/", index_page)
    app.router.add_get("/v1/health", health_check)
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/audio/speech", audio_speech)
    app.router.add_post("/v1/audio/speech-stream", audio_speech_stream)
    # Serve static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.router.add_static("/static/", path=str(static_path), name="static")
    return app


async def main():
    global tts_engine

    parser = argparse.ArgumentParser(description="Higgs Audio v3 TTS API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8081, help="Port number")
    parser.add_argument("--model-path", default=os.environ.get(
        "HIGGS_MODEL_PATH", "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"
    ), help="Model directory path")
    parser.add_argument("--device", default=os.environ.get(
        "HIGGS_DEVICE", "cuda:0"
    ), help="GPU device (e.g. cuda:0)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"\n{'='*60}")
    print(f"  Higgs Audio v3 TTS — API Server")
    print(f"{'='*60}")
    print(f"  Model:  {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Dtype:  {args.dtype}")
    print(f"  Host:   {args.host}:{args.port}")
    print(f"{'='*60}\n")

    tts_engine = HiggsTTS(model_path=args.model_path, device=args.device, dtype=dtype)
    tts_engine.get_vram_info()

    print(f"\n  Server ready at http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print(f"  Web UI:    http://localhost:{args.port}/")
    print(f"  Health:    http://localhost:{args.port}/v1/health")
    print(f"  API docs:  See readme.md\n")

    app = create_app(args)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
