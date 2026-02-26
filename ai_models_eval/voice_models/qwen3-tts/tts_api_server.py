#!/usr/bin/env python3
"""OpenAI-compatible TTS API server built on top of Qwen3-TTS.

Endpoint: POST /v1/audio/speech
Compatible with the OpenAI TTS API format, with extra fields for voice cloning.

Supported response formats:
  - pcm   : raw PCM16LE bytes (streamed as chunked response)
  - wav   : WAV file (streamed with header; data length set to 0xFFFFFFFF)
  - mp3   : MP3 bytes (requires pydub + ffmpeg; full response after generation)

Usage (voice clone / Base model):
  python3 tts_api_server.py \
    --host 0.0.0.0 \
    --port 8090 \
    --voice-clone-model-id models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main

Voice clone request (extra fields beyond OpenAI spec):
  {
    "model": "tts-1",
    "input": "Text to synthesize",
    "voice": "alloy",
    "response_format": "wav",
    "reference_audio": "<base64-encoded WAV bytes>",
    "reference_text": "Transcript of the reference audio"
  }
  When reference_audio + reference_text are present, voice_clone mode is used.
  Otherwise, custom_voice mode is used (requires CustomVoice model).

Voice mapping (OpenAI → Qwen3):
  OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) are all
  mapped to the default Qwen3 speaker "vivian".  You can also pass any
  Qwen3-supported speaker name directly in the 'voice' field.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import struct
import threading
from typing import Optional

from aiohttp import web

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
logger = logging.getLogger("tts_api_server")

# OpenAI voice names → Qwen3 speaker (fallback mapping)
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "vivian",
    "echo": "vivian",
    "fable": "vivian",
    "onyx": "vivian",
    "nova": "vivian",
    "shimmer": "vivian",
}

SUPPORTED_FORMATS = {"pcm", "wav", "mp3"}


def _wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16, data_size: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV RIFF header.  data_size=0xFFFFFFFF enables streaming."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    # chunk_size in RIFF header = 36 + data_size; cap at 0xFFFFFFFF for streaming
    chunk_size = min(data_size + 36, 0xFFFFFFFF)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


async def health_handler(request: web.Request) -> web.Response:
    service: QwenStreamingService = request.app["service"]
    return web.json_response(
        {
            "status": "ok",
            "models_loaded": {
                "custom_voice": "custom_voice" in service._models,
                "voice_clone": "voice_clone" in service._models,
            },
            "model_ids": service.model_ids,
            "sampling_rates": {
                "custom_voice": service.sampling_rate_for_mode("custom_voice"),
                "voice_clone": service.sampling_rate_for_mode("voice_clone"),
            },
        }
    )


async def models_handler(request: web.Request) -> web.Response:
    """List available models (OpenAI-compatible)."""
    return web.json_response(
        {
            "object": "list",
            "data": [
                {"id": "tts-1", "object": "model", "owned_by": "qwen"},
                {"id": "tts-1-hd", "object": "model", "owned_by": "qwen"},
            ],
        }
    )


async def speech_handler(request: web.Request) -> web.Response:
    """POST /v1/audio/speech — OpenAI-compatible TTS endpoint.

    Extra fields (beyond OpenAI spec) for voice cloning:
      reference_audio : base64-encoded WAV bytes of the reference speaker
      reference_text  : transcript of the reference audio
    When both are supplied, voice_clone mode is used automatically.
    """
    service: QwenStreamingService = request.app["service"]
    stop_event: Optional[threading.Event] = None
    stream_iter = None

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}}, status=400)

    text: str = str(body.get("input", "")).strip()
    if not text:
        return web.json_response(
            {"error": {"message": "'input' field is required and must be non-empty.", "type": "invalid_request_error"}},
            status=400,
        )

    voice: str = str(body.get("voice", DEFAULT_SPEAKER)).strip() or DEFAULT_SPEAKER
    response_format: str = str(body.get("response_format", "wav")).strip().lower()
    # model and speed fields are accepted but not currently applied
    _speed: float = float(body.get("speed", 1.0))

    # Voice-clone extension fields
    reference_audio_b64: str = str(body.get("reference_audio", "")).strip()
    reference_text: str = str(body.get("reference_text", "")).strip()

    if response_format not in SUPPORTED_FORMATS:
        return web.json_response(
            {
                "error": {
                    "message": f"Unsupported response_format '{response_format}'. Supported: {sorted(SUPPORTED_FORMATS)}",
                    "type": "invalid_request_error",
                }
            },
            status=400,
        )

    # Decide mode: voice_clone when reference audio + text are both provided
    reference_audio_bytes: Optional[bytes] = None
    if reference_audio_b64 and reference_text:
        try:
            reference_audio_bytes = base64.b64decode(reference_audio_b64)
        except Exception:
            return web.json_response(
                {"error": {"message": "reference_audio must be valid base64-encoded audio bytes.", "type": "invalid_request_error"}},
                status=400,
            )
        mode = "voice_clone"
        qwen_speaker = DEFAULT_SPEAKER  # not used in voice_clone mode
    else:
        mode = "custom_voice"
        # Map OpenAI voice name to Qwen3 speaker
        qwen_speaker = OPENAI_VOICE_MAP.get(voice.lower(), voice)

    try:
        model = await service.ensure_model(mode)
        if mode == "custom_voice":
            qwen_speaker = service.normalize_speaker(qwen_speaker, model)
    except ValueError as exc:
        return web.json_response(
            {"error": {"message": str(exc), "type": "invalid_request_error"}},
            status=400,
        )
    except Exception as exc:
        logger.exception("Model load failed")
        return web.json_response(
            {"error": {"message": f"Model load failed: {exc}", "type": "server_error"}},
            status=500,
        )

    params = StreamParams(
        text=text,
        mode=mode,
        language=DEFAULT_LANGUAGE,
        speaker=qwen_speaker,
        instruction="",
        reference_audio_bytes=reference_audio_bytes,
        reference_text=reference_text,
    )

    sample_rate = service.sampling_rate_for_mode(mode)
    stop_event = threading.Event()
    service.register_active_stream(mode, stop_event)
    stream_iter = service.stream_pcm16(params, stop_event=stop_event)

    if response_format == "pcm":
        # Stream raw PCM16LE bytes
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "audio/pcm",
                "Cache-Control": "no-cache",
                "X-Sample-Rate": str(sample_rate),
                "X-Audio-Format": "pcm16le",
                "X-Channels": "1",
            },
        )
        await response.prepare(request)
        try:
            async for chunk in stream_iter:
                if stop_event.is_set():
                    break
                try:
                    await response.write(chunk)
                except ConnectionResetError:
                    logger.info("Client disconnected during PCM stream.")
                    stop_event.set()
                    break
        finally:
            stop_event.set()
            service.clear_active_stream(mode, stop_event)
            try:
                await stream_iter.aclose()
            except Exception:
                pass
        try:
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    elif response_format == "wav":
        # Stream WAV: send header first, then PCM chunks
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "audio/wav",
                "Cache-Control": "no-cache",
                "X-Sample-Rate": str(sample_rate),
            },
        )
        await response.prepare(request)
        # Write WAV header with unknown data length (streaming mode)
        try:
            await response.write(_wav_header(sample_rate))
        except ConnectionResetError:
            stop_event.set()
            service.clear_active_stream(mode, stop_event)
            return response

        try:
            async for chunk in stream_iter:
                if stop_event.is_set():
                    break
                try:
                    await response.write(chunk)
                except ConnectionResetError:
                    logger.info("Client disconnected during WAV stream.")
                    stop_event.set()
                    break
        finally:
            stop_event.set()
            service.clear_active_stream(mode, stop_event)
            try:
                await stream_iter.aclose()
            except Exception:
                pass
        try:
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    else:
        # mp3: collect all PCM, encode with pydub, return in one response
        try:
            import pydub  # noqa: F401
        except ImportError:
            stop_event.set()
            service.clear_active_stream(mode, stop_event)
            try:
                await stream_iter.aclose()
            except Exception:
                pass
            return web.json_response(
                {"error": {"message": "MP3 encoding requires 'pydub' and 'ffmpeg'. Install them or use 'wav'/'pcm'.", "type": "server_error"}},
                status=500,
            )

        pcm_data = bytearray()
        try:
            async for chunk in stream_iter:
                if stop_event.is_set():
                    break
                pcm_data.extend(chunk)
        finally:
            stop_event.set()
            service.clear_active_stream(mode, stop_event)
            try:
                await stream_iter.aclose()
            except Exception:
                pass

        try:
            from pydub import AudioSegment
            audio = AudioSegment(
                data=bytes(pcm_data),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )
            mp3_buf = io.BytesIO()
            audio.export(mp3_buf, format="mp3")
            mp3_bytes = mp3_buf.getvalue()
        except Exception as exc:
            logger.exception("MP3 encoding failed")
            return web.json_response(
                {"error": {"message": f"MP3 encoding failed: {exc}", "type": "server_error"}},
                status=500,
            )

        return web.Response(
            body=mp3_bytes,
            content_type="audio/mpeg",
            headers={"Cache-Control": "no-cache"},
        )


def build_app(
    custom_model_id: str,
    voice_clone_model_id: str,
    device: str,
    dtype: str,
    attn_implementation: str,
) -> web.Application:
    app = web.Application(client_max_size=10 * 1024 * 1024)
    app["service"] = QwenStreamingService(
        custom_model_id=custom_model_id,
        voice_clone_model_id=voice_clone_model_id,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    app.router.add_get("/health", health_handler)
    app.router.add_get("/v1/models", models_handler)
    app.router.add_post("/v1/audio/speech", speech_handler)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible Qwen3-TTS API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--custom-model-id",
        default=os.environ.get(
            "QWEN_TTS_CUSTOM_MODEL_ID",
            os.environ.get("QWEN_TTS_MODEL_ID", DEFAULT_CUSTOM_MODEL_ID),
        ),
        help="Model path for custom_voice mode (named speakers)",
    )
    parser.add_argument(
        "--voice-clone-model-id",
        default=os.environ.get("QWEN_TTS_VOICE_CLONE_MODEL_ID", DEFAULT_VOICE_CLONE_MODEL_ID),
        help="Model path for voice_clone mode (reference audio + text). Use the Base model here.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get(
            "QWEN_TTS_DEVICE",
            "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu",
        ),
    )
    parser.add_argument("--dtype", default=os.environ.get("QWEN_TTS_DTYPE", "bfloat16"))
    parser.add_argument("--attn-implementation", default=os.environ.get("QWEN_TTS_ATTN_IMPL", "flash_attention_2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting OpenAI-compatible TTS API server on %s:%s", args.host, args.port)
    logger.info("  custom_voice model  : %s", args.custom_model_id)
    logger.info("  voice_clone model   : %s", args.voice_clone_model_id)
    logger.info("  Device: %s  |  dtype: %s  |  attn: %s", args.device, args.dtype, args.attn_implementation)
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
