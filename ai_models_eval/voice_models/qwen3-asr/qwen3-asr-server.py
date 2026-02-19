#!/usr/bin/env python3
"""
Qwen3-ASR Streaming Backend Service
=====================================
aiohttp + transformers TextIteratorStreamer for real token-by-token streaming.

Endpoints
---------
GET  /              - Web UI (index.html)
GET  /health        - Health check
POST /api/transcribe         - Upload audio file → full transcription (JSON)
POST /api/transcribe/stream  - Upload audio file → stream tokens (SSE)
GET  /ws            - WebSocket for real-time mic streaming
"""

import asyncio
import io
import json
import logging
import os
import ssl
import subprocess
import sys
from threading import Thread
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from aiohttp import web
import aiohttp

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(SCRIPT_DIR, "repos", "Qwen3-ASR")
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "models", "tts_hf_models", "Qwen", "Qwen3-ASR-1.7B_main",
)

sys.path.insert(0, REPO_PATH)

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    TextIteratorStreamer,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen3-asr")

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_SR = 16_000          # model input sample rate
MIN_AUDIO_SEC = 0.5         # minimum audio length to process
HOST = "0.0.0.0"
PORT = 8765

# ── globals (set in startup) ──────────────────────────────────────────────────
hf_model = None     # Qwen3ASRForConditionalGeneration
processor = None    # Qwen3ASRProcessor


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_sync() -> None:
    global hf_model, processor
    logger.info("Loading Qwen3-ASR-1.7B from: %s", MODEL_PATH)
    hf_model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
    logger.info("Model loaded ✓  device=%s  dtype=%s", hf_model.device, hf_model.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  Audio helpers
# ═══════════════════════════════════════════════════════════════════════════════

def decode_audio_file(data: bytes) -> np.ndarray:
    """Decode an uploaded audio file (wav / mp3 / webm …) to float32 mono @ 16 kHz."""
    try:
        with io.BytesIO(data) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
    except Exception:
        with io.BytesIO(data) as f:
            wav, sr = librosa.load(f, sr=None, mono=True, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav.astype(np.float32)


def resample_pcm(pcm: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample raw float32 PCM from src_sr → TARGET_SR (no-op if already correct)."""
    if src_sr == TARGET_SR:
        return pcm
    return librosa.resample(pcm, orig_sr=src_sr, target_sr=TARGET_SR).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(context: str = "", language: Optional[str] = None) -> str:
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if language:
        base += f"language {language}<asr_text>"
    return base


def prepare_inputs(wav: np.ndarray, prompt: str) -> dict:
    """Tokenise + extract audio features; move to GPU with correct dtypes."""
    batch = processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    result = {}
    for k, v in batch.items():
        v = v.to(hf_model.device)
        if v.dtype.is_floating_point:
            v = v.to(hf_model.dtype)
        result[k] = v
    return result


def _streaming_worker(
    wav: np.ndarray,
    language: Optional[str],
    context: str,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
) -> None:
    """
    Run in a thread-pool executor.
    Spawns model.generate() in a sub-thread (via TextIteratorStreamer),
    then pushes ('token', text) / ('done', None) / ('error', msg) into the asyncio queue.
    """
    try:
        prompt = build_prompt(context=context, language=language)
        inputs = prepare_inputs(wav, prompt)

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
            clean_up_tokenization_spaces=False,
        )

        gen_thread = Thread(
            target=hf_model.generate,
            kwargs={**inputs, "max_new_tokens": 512, "streamer": streamer},
            daemon=True,
        )
        gen_thread.start()

        for token_text in streamer:
            if token_text:
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token_text))

        gen_thread.join()
        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    except Exception as exc:
        logger.exception("streaming_worker error")
        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))


async def run_streaming(
    wav: np.ndarray,
    language: Optional[str],
    context: str,
) -> asyncio.Queue:
    """
    Launch streaming generation in a thread pool and return the asyncio queue.
    Caller must drain the queue until it receives ('done', …) or ('error', …).
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    loop.run_in_executor(None, _streaming_worker, wav, language, context, loop, queue)
    return queue


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP handlers
# ═══════════════════════════════════════════════════════════════════════════════

async def handle_index(request: web.Request) -> web.Response:
    return web.FileResponse(os.path.join(SCRIPT_DIR, "index.html"))


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "model": "Qwen3-ASR-1.7B"})


async def handle_transcribe(request: web.Request) -> web.Response:
    """POST /api/transcribe  — upload audio, return complete JSON transcription."""
    try:
        data = await request.post()
        audio_file = data.get("audio")
        language = (data.get("language") or "").strip() or None
        context = (data.get("context") or "").strip()

        if not audio_file:
            return web.json_response({"error": "Missing 'audio' field"}, status=400)

        audio_bytes = audio_file.file.read()
        wav = decode_audio_file(audio_bytes)

        # Collect all streamed tokens
        queue = await run_streaming(wav, language, context)
        tokens = []
        while True:
            kind, val = await queue.get()
            if kind == "done":
                break
            elif kind == "token":
                tokens.append(val)
            elif kind == "error":
                return web.json_response({"error": val}, status=500)

        return web.json_response({"text": "".join(tokens)})

    except Exception as exc:
        logger.exception("handle_transcribe error")
        return web.json_response({"error": str(exc)}, status=500)


async def handle_transcribe_stream(request: web.Request) -> web.StreamResponse:
    """POST /api/transcribe/stream  — upload audio, stream tokens via SSE."""
    try:
        data = await request.post()
        audio_file = data.get("audio")
        language = (data.get("language") or "").strip() or None
        context = (data.get("context") or "").strip()

        if not audio_file:
            return web.json_response({"error": "Missing 'audio' field"}, status=400)

        audio_bytes = audio_file.file.read()
        wav = decode_audio_file(audio_bytes)

        resp = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",
            }
        )
        await resp.prepare(request)

        queue = await run_streaming(wav, language, context)
        while True:
            kind, val = await queue.get()
            if kind == "done":
                await resp.write(b"data: [DONE]\n\n")
                break
            elif kind == "token":
                payload = json.dumps({"token": val})
                await resp.write(f"data: {payload}\n\n".encode())
            elif kind == "error":
                payload = json.dumps({"error": val})
                await resp.write(f"data: {payload}\n\n".encode())
                break

        return resp

    except Exception as exc:
        logger.exception("handle_transcribe_stream error")
        return web.json_response({"error": str(exc)}, status=500)


# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket handler — real-time mic streaming
# ═══════════════════════════════════════════════════════════════════════════════

async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    """
    WebSocket protocol
    ------------------
    Client → Server (TEXT, JSON):
        { cmd: "config", sampleRate: <int>, language: <str|null>, context: <str> }
        { cmd: "flush" } -- VAD detected a pause; process buffered audio and stream result
        { cmd: "end" }   -- recording stopped; flush remaining buffer
        { cmd: "clear" } -- discard buffered audio without inference

    Client → Server (BINARY):
        Raw float32 little-endian PCM mono at sampleRate.
        Only sent during detected voice activity (VAD gated on the client).

    Server → Client (TEXT, JSON):
        { type: "config_ok" }
        { type: "token",      text:    <str> }
        { type: "sentence_done" }
        { type: "end_ok" }
        { type: "cleared" }
        { type: "error",      message: <str> }
    """
    ws = web.WebSocketResponse(max_msg_size=50 * 1024 * 1024)
    await ws.prepare(request)
    logger.info("WS connected: %s", request.remote)

    # Per-connection state
    src_sr: int = TARGET_SR
    language: Optional[str] = None
    context: str = ""
    pcm_accum: np.ndarray = np.zeros((0,), dtype=np.float32)
    inferring: bool = False

    async def do_inference(wav_chunk: np.ndarray) -> None:
        nonlocal inferring
        inferring = True
        try:
            wav_16k = resample_pcm(wav_chunk, src_sr)
            queue = await run_streaming(wav_16k, language, context)
            while True:
                kind, val = await queue.get()
                if kind == "done":
                    await ws.send_json({"type": "sentence_done"})
                    break
                elif kind == "token":
                    await ws.send_json({"type": "token", "text": val})
                elif kind == "error":
                    await ws.send_json({"type": "error", "message": val})
                    break
        except Exception as exc:
            logger.exception("do_inference error")
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            inferring = False

    async def flush_buffer() -> None:
        """Process whatever audio is in pcm_accum, then clear it."""
        nonlocal pcm_accum
        if len(pcm_accum) < src_sr * MIN_AUDIO_SEC:
            pcm_accum = np.zeros((0,), dtype=np.float32)
            return
        segment = pcm_accum.copy()
        pcm_accum = np.zeros((0,), dtype=np.float32)
        await do_inference(segment)

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                cmd = data.get("cmd", "")

                if cmd == "config":
                    src_sr = int(data.get("sampleRate", TARGET_SR))
                    language = (data.get("language") or "").strip() or None
                    context = (data.get("context") or "").strip()
                    pcm_accum = np.zeros((0,), dtype=np.float32)
                    logger.info("WS config: sr=%d lang=%s", src_sr, language)
                    await ws.send_json({"type": "config_ok"})

                elif cmd == "flush":
                    # VAD-triggered sentence boundary — run inference if not already running
                    if not inferring:
                        await flush_buffer()

                elif cmd == "end":
                    # Recording stopped — flush remainder
                    if not inferring:
                        await flush_buffer()
                    await ws.send_json({"type": "end_ok"})

                elif cmd == "clear":
                    pcm_accum = np.zeros((0,), dtype=np.float32)
                    await ws.send_json({"type": "cleared"})

            elif msg.type == aiohttp.WSMsgType.BINARY:
                # Voice-active PCM from client; just accumulate
                chunk = np.frombuffer(msg.data, dtype=np.float32).copy()
                pcm_accum = np.concatenate([pcm_accum, chunk])

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WS protocol error: %s", ws.exception())
                break

    except Exception:
        logger.exception("WS handler error")

    logger.info("WS disconnected: %s", request.remote)
    return ws


# ═══════════════════════════════════════════════════════════════════════════════
#  App factory + startup
# ═══════════════════════════════════════════════════════════════════════════════

def make_app() -> web.Application:
    app = web.Application()

    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/api/transcribe", handle_transcribe)
    app.router.add_post("/api/transcribe/stream", handle_transcribe_stream)
    app.router.add_get("/ws", handle_websocket)

    async def on_startup(app: web.Application) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_model_sync)

    app.on_startup.append(on_startup)
    return app


# ═══════════════════════════════════════════════════════════════════════════════
#  SSL — auto-generate a self-signed certificate (required for Chrome mic access)
# ═══════════════════════════════════════════════════════════════════════════════

SSL_DIR  = os.path.join(SCRIPT_DIR, "ssl")
CERT_PEM = os.path.join(SSL_DIR, "cert.pem")
KEY_PEM  = os.path.join(SSL_DIR, "key.pem")


def ensure_ssl_cert() -> ssl.SSLContext:
    """Generate a self-signed cert (if absent) and return an SSLContext."""
    os.makedirs(SSL_DIR, exist_ok=True)

    if not (os.path.exists(CERT_PEM) and os.path.exists(KEY_PEM)):
        logger.info("Generating self-signed SSL certificate → %s", SSL_DIR)
        # Use a temporary openssl config file so we can embed a SAN —
        # Chrome rejects certs that lack a Subject Alternative Name.
        cfg_path = os.path.join(SSL_DIR, "openssl.cnf")
        cfg_content = (
            "[req]\n"
            "distinguished_name = req_distinguished_name\n"
            "x509_extensions    = v3_req\n"
            "prompt             = no\n"
            "[req_distinguished_name]\n"
            "CN = localhost\n"
            "[v3_req]\n"
            "subjectAltName = @alt_names\n"
            "[alt_names]\n"
            "DNS.1 = localhost\n"
            "IP.1  = 127.0.0.1\n"
        )
        with open(cfg_path, "w") as f:
            f.write(cfg_content)

        subprocess.run(
            [
                "openssl", "req", "-x509",
                "-newkey", "rsa:2048",
                "-keyout", KEY_PEM,
                "-out",    CERT_PEM,
                "-days",   "3650",
                "-nodes",
                "-config", cfg_path,
            ],
            check=True,
            capture_output=True,
        )
        logger.info("SSL cert generated ✓")

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_PEM, KEY_PEM)
    return ctx


if __name__ == "__main__":
    ssl_ctx = ensure_ssl_cert()
    app = make_app()
    logger.info("Starting HTTPS server on https://%s:%d", HOST, PORT)
    logger.info("Open in Chrome: https://localhost:%d", PORT)
    logger.info("(Accept the self-signed cert warning once, then mic will work)")
    web.run_app(app, host=HOST, port=PORT, ssl_context=ssl_ctx, print=logger.info)
