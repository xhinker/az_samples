"""
indextts-server.py - WebUI server for IndexTTS2 (aiohttp)

Serves a pure HTML/CSS/JS web interface that lets users:
  - Paste text into an input box
  - Upload a reference voice (WAV/MP3/etc.)
  - Stream synthesised speech in real time via the Web Audio API

Endpoints:
  GET  /                    - serve the WebUI
  POST /api/upload-voice    - upload a reference voice, returns {voice_id}
  POST /api/tts/stream      - streaming TTS (chunked WAV)
  GET  /api/health          - health check

Run:
  ./indextts-venv-py311/bin/python indextts-server.py [--host 0.0.0.0] [--port 7860]
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from aiohttp import web

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from audio_gen import SAMPLE_RATE, get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR  = os.path.join(BASE_DIR, "static")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")

os.makedirs(STATIC_DIR,  exist_ok=True)
os.makedirs(UPLOAD_DIR,  exist_ok=True)

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

routes = web.RouteTableDef()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@routes.get("/")
async def index(request: web.Request) -> web.Response:
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return web.Response(text=content, content_type="text/html")


@routes.post("/api/upload-voice")
async def upload_voice(request: web.Request) -> web.Response:
    """
    Upload a reference voice file.

    Multipart field:
      file – audio file (WAV, MP3, FLAC, OGG, M4A, AAC)

    Returns:
      { "voice_id": "<uuid>", "filename": "<uuid>.<ext>" }
    """
    reader = await request.multipart()
    file_data = None
    file_ext = ".wav"

    async for field in reader:
        if field.name == "file":
            if field.filename:
                ext = os.path.splitext(field.filename)[1].lower()
                file_ext = ext if ext in SUPPORTED_EXTS else ".wav"
            file_data = await field.read()

    if not file_data:
        raise web.HTTPBadRequest(text="No file uploaded")

    voice_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{voice_id}{file_ext}")
    with open(save_path, "wb") as f:
        f.write(file_data)

    logger.info("Reference voice uploaded: %s", save_path)
    return web.json_response(
        {"voice_id": voice_id, "filename": f"{voice_id}{file_ext}"}
    )


@routes.post("/api/tts/stream")
async def stream_tts(request: web.Request) -> web.StreamResponse:
    """
    Streaming TTS endpoint.

    Request JSON:
    {
        "text":     "text to synthesise",
        "voice_id": "<uuid returned by /api/upload-voice>"
    }

    Response: chunked WAV (44-byte header + raw PCM int16 chunks).
    Each chunk is emitted as soon as a text segment has been decoded,
    enabling real-time playback with minimal latency.
    """
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON")

    text     = body.get("text", "").strip()
    voice_id = body.get("voice_id", "").strip()

    if not text:
        raise web.HTTPBadRequest(text="'text' field is required")
    if not voice_id:
        raise web.HTTPBadRequest(text="'voice_id' field is required")

    # Locate the uploaded voice file
    voice_path = None
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(UPLOAD_DIR, f"{voice_id}{ext}")
        if os.path.exists(candidate):
            voice_path = candidate
            break

    if voice_path is None:
        raise web.HTTPNotFound(text=f"Voice '{voice_id}' not found. Upload it first.")

    engine = request.app["engine"]

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
        logger.info(
            "TTS stream start: voice=%s text_len=%d", voice_id, len(text)
        )
        async for chunk in engine.stream_wav(text, voice_path):
            await resp.write(chunk)
    except Exception:
        logger.exception("TTS streaming error")
    finally:
        await resp.write_eof()

    return resp


@routes.get("/api/health")
async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


# ---------------------------------------------------------------------------
# App factory & startup
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application):
    logger.info("Loading IndexTTS2 model …")
    engine = get_engine(use_fp16=True, use_cuda_kernel=False)
    app["engine"] = engine
    await engine.load_model()
    logger.info("IndexTTS WebUI server is ready")


def create_app() -> web.Application:
    app = web.Application(client_max_size=200 * 1024 * 1024)  # 200 MB uploads
    app.add_routes(routes)
    app.router.add_static("/static", STATIC_DIR, name="static")
    app.on_startup.append(on_startup)
    return app


def main():
    parser = argparse.ArgumentParser(description="IndexTTS WebUI Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        logger.error(
            "WebUI not found at %s — please make sure static/index.html exists.", html_path
        )
        sys.exit(1)

    app = create_app()
    logger.info("IndexTTS WebUI → http://%s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
