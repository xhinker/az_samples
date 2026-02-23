"""
indextts-api.py - OpenAI-compatible TTS HTTP API (aiohttp)

Compatible with the OpenAI Audio Speech API format:
  POST /v1/audio/speech

Additional endpoints:
  GET  /v1/models              - list models
  GET  /v1/voices              - list registered voices
  POST /v1/voices              - upload / register a reference voice
  POST /v1/audio/speech/stream - always-streaming variant
  GET  /health                 - health check

Reference voice management:
  Voices are stored as audio files under the `voices/` directory.
  The `voice` field in the request body maps to `voices/<voice>.<ext>`.
  A default voice can be set by placing a file at `voices/default.wav`.

Run:
  ./indextts-venv-py311/bin/python indextts-api.py [--host 0.0.0.0] [--port 8880]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from aiohttp import web

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from audio_gen import SAMPLE_RATE, build_streaming_wav_header, get_engine, pcm_to_wav_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VOICES_DIR = os.path.join(BASE_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

routes = web.RouteTableDef()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_voice(voice_name: str) -> str | None:
    """Return the absolute path of a registered voice file, or None."""
    for ext in SUPPORTED_AUDIO_EXTS:
        p = os.path.join(VOICES_DIR, f"{voice_name}{ext}")
        if os.path.exists(p):
            return p
    return None


def _sanitize_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@routes.get("/health")
async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "model": "indextts-2"})


@routes.get("/v1/models")
async def list_models(request: web.Request) -> web.Response:
    """OpenAI-style model listing."""
    return web.json_response({
        "object": "list",
        "data": [
            {"id": "tts-1",       "object": "model", "owned_by": "IndexTeam"},
            {"id": "tts-1-hd",    "object": "model", "owned_by": "IndexTeam"},
            {"id": "indextts-2",  "object": "model", "owned_by": "IndexTeam"},
        ],
    })


@routes.get("/v1/voices")
async def list_voices(request: web.Request) -> web.Response:
    voices = []
    for f in Path(VOICES_DIR).iterdir():
        if f.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            voices.append({"id": f.stem, "name": f.stem, "file": f.name})
    return web.json_response({"voices": voices})


@routes.post("/v1/voices")
async def upload_voice(request: web.Request) -> web.Response:
    """
    Register a reference voice.

    Multipart fields:
      name  – voice identifier (alphanumeric / dash / underscore)
      file  – audio file
    """
    reader = await request.multipart()
    voice_name = None
    file_data = None
    file_ext = ".wav"

    async for field in reader:
        if field.name == "name":
            voice_name = (await field.read()).decode().strip()
        elif field.name == "file":
            if field.filename:
                file_ext = os.path.splitext(field.filename)[1].lower() or ".wav"
            file_data = await field.read()

    if not voice_name:
        raise web.HTTPBadRequest(text="Missing 'name' field")
    if not file_data:
        raise web.HTTPBadRequest(text="Missing 'file' field")

    voice_name = _sanitize_name(voice_name)
    if not voice_name:
        raise web.HTTPBadRequest(text="Invalid voice name")

    save_path = os.path.join(VOICES_DIR, f"{voice_name}{file_ext}")
    with open(save_path, "wb") as f:
        f.write(file_data)

    logger.info("Voice '%s' saved to %s", voice_name, save_path)
    return web.json_response(
        {"id": voice_name, "name": voice_name, "file": f"{voice_name}{file_ext}"},
        status=201,
    )


@routes.post("/v1/audio/speech")
async def create_speech(request: web.Request) -> web.Response:
    """
    OpenAI-compatible TTS endpoint.

    Request body (JSON):
    {
        "model":           "tts-1",     // accepted but ignored; always IndexTTS2
        "input":           "text …",    // required
        "voice":           "alloy",     // maps to voices/<voice>.<ext>
        "response_format": "wav",       // "wav" or "pcm"  (default: wav)
        "speed":           1.0,         // accepted but ignored
        "stream":          false        // true → chunked streaming response
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    text = body.get("input", "").strip()
    if not text:
        raise web.HTTPBadRequest(text="'input' field is required and must not be empty")

    voice_name = body.get("voice", "default")
    response_format = body.get("response_format", "wav").lower()
    do_stream = body.get("stream", False)

    voice_path = _find_voice(voice_name)
    if voice_path is None:
        raise web.HTTPNotFound(
            text=(
                f"Voice '{voice_name}' not found. "
                f"Register it via POST /v1/voices or place an audio file in {VOICES_DIR}/"
            )
        )

    engine: object = request.app["engine"]

    if do_stream:
        return await _streaming_response(request, engine, text, voice_path)

    # Non-streaming: collect all PCM then return
    try:
        wav_bytes = await engine.generate_wav(text, voice_path)
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise web.HTTPInternalServerError(text=str(exc))

    if response_format == "pcm":
        return web.Response(
            body=wav_bytes[44:],   # strip the 44-byte WAV header
            content_type="audio/pcm",
            headers={"X-Sample-Rate": str(SAMPLE_RATE)},
        )

    return web.Response(body=wav_bytes, content_type="audio/wav")


@routes.post("/v1/audio/speech/stream")
async def create_speech_stream(request: web.Request) -> web.StreamResponse:
    """
    Always-streaming variant of /v1/audio/speech.

    Same request format; always returns chunked WAV.
    """
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON body")

    text = body.get("input", "").strip()
    if not text:
        raise web.HTTPBadRequest(text="'input' field is required")

    voice_name = body.get("voice", "default")
    voice_path = _find_voice(voice_name)
    if voice_path is None:
        raise web.HTTPNotFound(text=f"Voice '{voice_name}' not found")

    engine = request.app["engine"]
    return await _streaming_response(request, engine, text, voice_path)


async def _streaming_response(
    request: web.Request,
    engine,
    text: str,
    voice_path: str,
) -> web.StreamResponse:
    """Internal helper: send a chunked streaming WAV response."""
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "audio/wav",
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "Access-Control-Allow-Origin": "*",
        },
    )
    await resp.prepare(request)
    try:
        async for chunk in engine.stream_wav(text, voice_path):
            await resp.write(chunk)
    except Exception:
        logger.exception("Error during streaming TTS")
    finally:
        await resp.write_eof()
    return resp


# ---------------------------------------------------------------------------
# App factory & startup
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application):
    logger.info("Loading IndexTTS2 model …")
    engine = get_engine(use_fp16=True, use_cuda_kernel=False)
    app["engine"] = engine
    await engine.load_model()
    logger.info("IndexTTS API server is ready")


def create_app() -> web.Application:
    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100 MB uploads
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    return app


def main():
    parser = argparse.ArgumentParser(description="IndexTTS OpenAI-compatible API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = create_app()
    logger.info("IndexTTS API → http://%s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
