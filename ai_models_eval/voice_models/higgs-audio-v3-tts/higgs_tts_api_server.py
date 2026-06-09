#!/usr/bin/env python3
"""OpenAI-compatible TTS API server for Higgs Audio v3 (transformers port).

Endpoints:
  GET  /v1/health          — health check
  GET  /v1/models          — list models (OpenAI-compatible)
  POST /v1/audio/speech    — TTS synthesis with real streaming

Streaming: real-time PCM/WAV chunked transfer (sub-second TTA).
Voice cloning: pass reference_audio (base64 WAV) + reference_text.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import base64
import io
import logging
import os
import struct
import threading
from pathlib import Path
from typing import Optional

from aiohttp import web

try:
    from .higgs_audio_gen import HiggsStreamingService, StreamParams
except ImportError:
    from higgs_audio_gen import HiggsStreamingService, StreamParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("higgs_tts_api")

SUPPORTED_FORMATS = {"pcm", "wav", "mp3"}

OPENAI_VOICE_MAP: dict[str, dict] = {
    "alloy": {"temperature": 0.8},
    "echo": {"temperature": 0.9},
    "fable": {"temperature": 0.7},
    "onyx": {"temperature": 0.6},
    "nova": {"temperature": 0.85},
    "shimmer": {"temperature": 0.75},
}


async def health_handler(request: web.Request) -> web.Response:
    service: HiggsStreamingService = request.app["service"]
    return web.json_response({
        "status": "ok",
        "model_loaded": service._model is not None,
        "model_path": service.model_path,
        "sample_rate": service.sample_rate,
        "device": service.device,
    })


async def models_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "owned_by": "boson"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "boson"},
            {"id": "higgs-audio-v3-tts", "object": "model", "owned_by": "boson"},
        ],
    })


async def _stream_pcm_chunks(
    service: HiggsStreamingService,
    params: StreamParams,
    stop_event: threading.Event,
    sid: int,
    response: web.StreamResponse,
) -> None:
    """Common streaming logic for WAV and PCM formats."""
    try:
        async for chunk in service.stream_pcm16(params, stop_event):
            if stop_event.is_set():
                break
            try:
                await response.write(chunk)
            except ConnectionResetError:
                logger.info("Client disconnected during stream.")
                stop_event.set()
                break
    finally:
        stop_event.set()
        service.clear_active_stream(sid)


async def speech_handler(request: web.Request) -> web.Response:
    """POST /v1/audio/speech — OpenAI-compatible TTS endpoint."""
    service: HiggsStreamingService = request.app["service"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status=400,
        )

    text: str = str(body.get("input", "")).strip()
    if not text:
        return web.json_response(
            {"error": {"message": "'input' field is required.", "type": "invalid_request_error"}},
            status=400,
        )

    voice: str = str(body.get("voice", "alloy")).strip().lower()
    response_format: str = str(body.get("response_format", "wav")).strip().lower()

    if response_format not in SUPPORTED_FORMATS:
        return web.json_response(
            {"error": {"message":
                        f"Unsupported format '{response_format}'. Options: {sorted(SUPPORTED_FORMATS)}",
             "type": "invalid_request_error"}},
            status=400,
        )

    # Voice cloning
    reference_audio_b64: str = str(body.get("reference_audio", "")).strip()
    reference_text: Optional[str] = body.get("reference_text")
    if reference_text is not None:
        reference_text = str(reference_text).strip() or None

    reference_audio_bytes: Optional[bytes] = None
    if reference_audio_b64:
        try:
            reference_audio_bytes = base64.b64decode(reference_audio_b64)
        except Exception:
            return web.json_response(
                {"error": {"message": "reference_audio must be valid base64 WAV bytes.",
                            "type": "invalid_request_error"}},
                status=400,
            )

    # Generation params
    voice_params = OPENAI_VOICE_MAP.get(voice, {"temperature": 0.8})
    temperature = float(body.get("temperature", voice_params.get("temperature", 0.8)))
    top_p = body.get("top_p")
    if top_p is not None:
        top_p = float(top_p)
    top_k = body.get("top_k")
    if top_k is not None:
        top_k = int(top_k)

    params = StreamParams(
        text=text,
        reference_audio_bytes=reference_audio_bytes,
        reference_sample_rate=24000,
        reference_text=reference_text,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    logger.info(
        "[TTS] fmt=%s voice=%s temp=%.2f ref=%s text=%r",
        response_format, voice, temperature,
        "yes" if reference_audio_bytes else "no", text[:80],
    )

    sample_rate = service.sample_rate
    stop_event = threading.Event()
    sid = service.register_active_stream(stop_event)

    try:
        await service.ensure_model()
    except Exception as exc:
        service.clear_active_stream(sid)
        logger.exception("Model load failed")
        return web.json_response(
            {"error": {"message": f"Model load failed: {exc}", "type": "server_error"}},
            status=500,
        )

    # ── WAV/PCM streaming (raw PCM16 — no header for browser playback) ──
    if response_format in ("wav", "pcm"):
        content_type = "audio/wav" if response_format == "wav" else "audio/pcm"
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": content_type,
                "Cache-Control": "no-cache",
                "X-Sample-Rate": str(sample_rate),
            },
        )
        await response.prepare(request)
        try:
            await _stream_pcm_chunks(service, params, stop_event, sid, response)
        except ConnectionResetError:
            stop_event.set()
            service.clear_active_stream(sid)
        finally:
            try:
                await response.write_eof()
            except ConnectionResetError:
                pass
        return response

    # ── MP3: collect all PCM, convert via ffmpeg ──
    else:
        try:
            all_pcm = b""
            async for chunk in service.stream_pcm16(params, stop_event):
                all_pcm += chunk

            if not all_pcm:
                return web.json_response(
                    {"error": {"message": "Generated empty audio.", "type": "server_error"}},
                    status=500,
                )

            import subprocess
            import tempfile

            # Build proper WAV file with correct data size
            wav_path = None
            mp3_path = None
            try:
                fd, wav_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                data_size = len(all_pcm)
                byte_rate = sample_rate * 2  # mono, 16-bit
                block_align = 2
                chunk_size = data_size + 36
                with open(wav_path, 'wb') as f:
                    f.write(struct.pack("<4sI4s4sIHHIIHH4sI",
                        b"RIFF", chunk_size, b"WAVE", b"fmt ",
                        16, 1, 1, sample_rate, byte_rate, block_align, 16,
                        b"data", data_size))
                    f.write(all_pcm)

                mp3_path = wav_path + ".mp3"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame",
                     "-qscale:a", "2", mp3_path],
                    capture_output=True, timeout=30, check=True,
                )
                with open(mp3_path, "rb") as f:
                    mp3_data = f.read()
            finally:
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
                if mp3_path and os.path.exists(mp3_path):
                    os.unlink(mp3_path)

        finally:
            stop_event.set()
            service.clear_active_stream(sid)

        return web.Response(
            body=mp3_data,
            content_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )


def build_app(model_path: str, device: str, dtype: str) -> web.Application:
    app = web.Application(client_max_size=30 * 1024 * 1024)
    app["service"] = HiggsStreamingService(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )

    app.router.add_get("/v1/health", health_handler)
    app.router.add_get("/v1/models", models_handler)
    app.router.add_post("/v1/audio/speech", speech_handler)

    # Serve static test page
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.router.add_get("/", lambda r: web.FileResponse(static_dir / "index.html"))
        app.router.add_static("/static", path=str(static_dir))

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Higgs Audio v3 TTS API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model-path", default=os.environ.get("HIGGS_MODEL_PATH", "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"))
    parser.add_argument("--device", default=os.environ.get("HIGGS_DEVICE", "cuda:2"), help="Target GPU (e.g. cuda:2 for RTX 3090). Model loads directly on this GPU only.")
    parser.add_argument("--dtype", default=os.environ.get("HIGGS_DTYPE", "bfloat16"))
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
    )
    logger.info("Starting Higgs Audio v3 TTS API on %s:%s (device=%s)",
                args.host, args.port, args.device)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
