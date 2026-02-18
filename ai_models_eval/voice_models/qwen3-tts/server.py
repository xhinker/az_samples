#!/usr/bin/env python3
"""Aiohttp server for Qwen3-TTS streaming (PCM16 over HTTP chunked response)."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path
from typing import AsyncIterator, Optional

from aiohttp import web
from aiohttp.web_request import FileField

try:
    from .audio_gen import (
        DEFAULT_CUSTOM_MODEL_ID,
        DEFAULT_LANGUAGE,
        DEFAULT_SPEAKER,
        DEFAULT_VOICE_CLONE_MODEL_ID,
        QwenStreamingService,
        StreamParams,
        torch,
    )
except ImportError:
    from audio_gen import (
        DEFAULT_CUSTOM_MODEL_ID,
        DEFAULT_LANGUAGE,
        DEFAULT_SPEAKER,
        DEFAULT_VOICE_CLONE_MODEL_ID,
        QwenStreamingService,
        StreamParams,
        torch,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("qwen3_tts_server")


async def index_handler(request: web.Request):
    static_dir = Path(request.app["static_dir"])
    return web.FileResponse(static_dir / "index.html")


async def health_handler(request: web.Request):
    service: QwenStreamingService = request.app["service"]
    custom_loaded = "custom_voice" in service._models
    clone_loaded = "voice_clone" in service._models
    return web.json_response(
        {
            "status": "ok",
            "models_loaded": {
                "custom_voice": custom_loaded,
                "voice_clone": clone_loaded,
            },
            "model_ids": service.model_ids,
            "sampling_rates": {
                "custom_voice": service.sampling_rate_for_mode("custom_voice"),
                "voice_clone": service.sampling_rate_for_mode("voice_clone"),
            },
        }
    )


async def stop_handler(request: web.Request):
    service: QwenStreamingService = request.app["service"]
    mode: Optional[str] = None
    try:
        if request.content_type and "application/json" in request.content_type:
            body = await request.json()
            mode = body.get("mode")
        else:
            form = await request.post()
            mode = form.get("mode")
    except Exception:
        mode = None

    if mode is not None:
        mode = str(mode).strip()
        if mode == "":
            mode = None
        elif mode not in {"custom_voice", "voice_clone"}:
            return web.json_response({"error": f"Unsupported mode: {mode!r}"}, status=400)

    stopped = service.request_stop(mode=mode)
    return web.json_response(
        {
            "status": "ok",
            "stopped_streams": stopped,
            "mode": mode or "all",
        }
    )


async def stream_handler(request: web.Request):
    service: QwenStreamingService = request.app["service"]
    response: Optional[web.StreamResponse] = None
    stream_iter: Optional[AsyncIterator[bytes]] = None
    stop_event: Optional[threading.Event] = None
    mode = "custom_voice"

    try:
        form = await request.post()
        text = str(form.get("text", "")).strip()
        mode = str(form.get("mode", "custom_voice")).strip()
        language = str(form.get("language", DEFAULT_LANGUAGE)).strip() or DEFAULT_LANGUAGE
        speaker = str(form.get("speaker", DEFAULT_SPEAKER)).strip() or DEFAULT_SPEAKER
        instruction = str(form.get("instruction", "")).strip()
        reference_text = str(form.get("reference_text", "")).strip()

        reference_audio_bytes = None
        reference_audio_field = form.get("reference_audio")
        if isinstance(reference_audio_field, FileField):
            reference_audio_bytes = reference_audio_field.file.read()

        if mode not in {"custom_voice", "voice_clone"}:
            raise ValueError(f"Unsupported mode: {mode!r}")
        if not text:
            raise ValueError("Text is required.")

        model = await service.ensure_model(mode)
        if mode == "custom_voice":
            speaker = service.normalize_speaker(speaker, model)

        params = StreamParams(
            text=text,
            mode=mode,
            language=language,
            speaker=speaker,
            instruction=instruction,
            reference_audio_bytes=reference_audio_bytes,
            reference_text=reference_text,
        )

        stop_event = threading.Event()
        service.register_active_stream(mode, stop_event)
        stream_iter = service.stream_pcm16(params, stop_event=stop_event)
        first_chunk = await anext(stream_iter)

        headers = {
            "Content-Type": "application/octet-stream",
            "Cache-Control": "no-cache",
            "X-Sample-Rate": str(service.sampling_rate_for_mode(mode)),
            "X-Audio-Format": "pcm16le",
        }
        response = web.StreamResponse(status=200, headers=headers)
        await response.prepare(request)
        await response.write(first_chunk)

        async for chunk_bytes in stream_iter:
            try:
                await response.write(chunk_bytes)
            except ConnectionResetError:
                logger.info("Client disconnected during stream.")
                stop_event.set()
                break

        try:
            await response.write_eof()
        except ConnectionResetError:
            logger.info("Client disconnected before EOF.")
        return response

    except StopAsyncIteration:
        return web.json_response({"error": "Model generated empty audio output."}, status=400)
    except Exception as exc:
        if stop_event is not None:
            stop_event.set()
        logger.exception("Streaming request failed")
        if response is not None:
            try:
                await response.write_eof()
            except Exception:
                pass
            return response
        return web.json_response(
            {"error": str(exc)},
            status=400,
        )
    finally:
        if stop_event is not None:
            stop_event.set()
            service.clear_active_stream(mode, stop_event)
        if stream_iter is not None:
            try:
                await stream_iter.aclose()
            except Exception:
                pass


def build_app(
    custom_model_id: str,
    voice_clone_model_id: str,
    device: str,
    dtype: str,
    attn_implementation: str,
) -> web.Application:
    static_dir = Path(__file__).parent / "static"
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    app = web.Application(client_max_size=30 * 1024 * 1024)
    app["static_dir"] = str(static_dir)
    app["service"] = QwenStreamingService(
        custom_model_id=custom_model_id,
        voice_clone_model_id=voice_clone_model_id,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    app.router.add_get("/", index_handler)
    app.router.add_get("/api/health", health_handler)
    app.router.add_post("/api/stop", stop_handler)
    app.router.add_post("/api/stream", stream_handler)
    app.router.add_static("/static", path=str(static_dir))
    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS streaming server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--custom-model-id",
        default=os.environ.get(
            "QWEN_TTS_CUSTOM_MODEL_ID",
            os.environ.get("QWEN_TTS_MODEL_ID", DEFAULT_CUSTOM_MODEL_ID),
        ),
    )
    parser.add_argument(
        "--model-id",
        dest="custom_model_id",
        help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-clone-model-id",
        default=os.environ.get(
            "QWEN_TTS_VOICE_CLONE_MODEL_ID",
            DEFAULT_VOICE_CLONE_MODEL_ID,
        ),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get(
            "QWEN_TTS_DEVICE",
            "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu",
        ),
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("QWEN_TTS_DTYPE", "bfloat16"),
    )
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("QWEN_TTS_ATTN_IMPL", "flash_attention_2"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(
        custom_model_id=args.custom_model_id,
        voice_clone_model_id=args.voice_clone_model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
