#!/usr/bin/env python3
"""Aiohttp server that streams LLM chat text and live TTS audio over WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import ssl
import subprocess
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiohttp
from aiohttp import ClientConnectionError, WSMsgType, web

try:
    from .audio_gen import (
        DEFAULT_CUSTOM_MODEL_ID,
        DEFAULT_LANGUAGE,
        DEFAULT_SPEAKER,
        DEFAULT_VOICE_CLONE_MODEL_ID,
        QwenStreamingService,
        StreamParams,
    )
except ImportError:
    from audio_gen import (
        DEFAULT_CUSTOM_MODEL_ID,
        DEFAULT_LANGUAGE,
        DEFAULT_SPEAKER,
        DEFAULT_VOICE_CLONE_MODEL_ID,
        QwenStreamingService,
        StreamParams,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("llm_audio_server")


DEFAULT_LLM_API_NAME = "macm4max_lmstudio"
DEFAULT_LLM_API_URL = "http://192.168.68.79:1234/v1"
DEFAULT_LLM_API_KEY = "lmstudio"
DEFAULT_LLM_MODEL_NAME = "qwen/qwen3-coder-next"
DEFAULT_ASR_URL = "https://localhost:8765"
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. Respond with plain spoken text only, "
    "without markdown, code fences, bullet lists, XML tags, JSON, or emojis. "
    "Write naturally so the response can be read aloud by TTS."
)

SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？;；:\n]+")

_SSL_DIR  = Path(__file__).parent / "ssl"
_CERT_PEM = _SSL_DIR / "cert.pem"
_KEY_PEM  = _SSL_DIR / "key.pem"


def _coerce_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _coerce_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _extract_text_from_delta_payload(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""

    chunks: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])

        text = choice.get("text")
        if isinstance(text, str):
            chunks.append(text)

    return "".join(chunks)


def _extract_ready_segments(buffer_text: str) -> tuple[list[str], str]:
    text = buffer_text
    segments: list[str] = []
    cursor = 0
    for match in SENTENCE_SPLIT_RE.finditer(text):
        end = match.end()
        candidate = text[cursor:end].strip()
        if len(candidate) >= 24:
            segments.append(candidate)
            cursor = end

    remainder = text[cursor:]
    if len(remainder) > 220:
        split_at = remainder.rfind(" ", 0, 200)
        if split_at < 80:
            split_at = 200
        head = remainder[:split_at].strip()
        if head:
            segments.append(head)
        remainder = remainder[split_at:]

    return segments, remainder


class LMStudioStreamingClient:
    def __init__(self, api_name: str, api_url: str, api_key: str, model_name: str):
        self.api_name = api_name
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session

        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                return self._session
            timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=None)
            self._session = aiohttp.ClientSession(timeout=timeout)
            return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        endpoint = f"{self.api_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        logger.info("Streaming from API=%s model=%s", self.api_name, self.model_name)

        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status >= 400:
                detail = (await response.text())[:2000]
                raise RuntimeError(
                    f"LLM API error {response.status} {response.reason}: {detail}"
                )

            async for raw_chunk in response.content:
                if not raw_chunk:
                    continue
                decoded = raw_chunk.decode("utf-8", errors="ignore")
                for line in decoded.splitlines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        return
                    try:
                        payload_json = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON stream line: %s", data[:120])
                        continue
                    text_delta = _extract_text_from_delta_payload(payload_json)
                    if text_delta:
                        yield text_delta


async def asr_transcribe_handler(request: web.Request) -> web.Response:
    """Proxy POST /api/asr/transcribe → Qwen3-ASR service /api/transcribe."""
    asr_url = request.app.get("asr_url", "")
    if not asr_url:
        return web.json_response({"error": "ASR service URL not configured"}, status=503)
    try:
        reader = await request.multipart()
        field = await reader.next()
        while field is not None and field.name != "audio":
            field = await reader.next()
        if field is None:
            return web.json_response({"error": "Missing 'audio' field"}, status=400)

        audio_data = await field.read()
        filename = field.filename or "voice.webm"
        content_type = field.headers.get("Content-Type", "audio/webm")

        form = aiohttp.FormData()
        form.add_field("audio", audio_data, filename=filename, content_type=content_type)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{asr_url}/api/transcribe", data=form, ssl=False
            ) as resp:
                result = await resp.json(content_type=None)
                return web.json_response(result, status=resp.status)
    except Exception as exc:
        logger.exception("ASR proxy error")
        return web.json_response({"error": str(exc)}, status=500)


async def index_handler(request: web.Request) -> web.FileResponse:
    static_dir = Path(request.app["static_dir"])
    return web.FileResponse(static_dir / "index.html")


async def health_handler(request: web.Request) -> web.Response:
    service: QwenStreamingService = request.app["service"]
    llm_client: LMStudioStreamingClient = request.app["llm_client"]
    ref_configured = request.app.get("ref_audio_bytes") is not None
    return web.json_response(
        {
            "status": "ok",
            "llm_api": {
                "api_name": llm_client.api_name,
                "api_url": llm_client.api_url,
                "model_name": llm_client.model_name,
            },
            "tts_models_loaded": {
                "custom_voice": "custom_voice" in service._models,
                "voice_clone": "voice_clone" in service._models,
            },
            "ref_audio_configured": ref_configured,
            "default_mode": "voice_clone" if ref_configured else "custom_voice",
        }
    )


async def _safe_send_json(ws: web.WebSocketResponse, payload: dict[str, Any]) -> bool:
    if ws.closed:
        return False
    try:
        await ws.send_json(payload)
        return True
    except (ConnectionResetError, ClientConnectionError):
        return False


async def _client_control_loop(
    ws: web.WebSocketResponse,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            msg = await ws.receive(timeout=0.5)
        except asyncio.TimeoutError:
            continue

        if msg.type == WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and str(payload.get("type")) == "stop":
                stop_event.set()
                return
            continue

        if msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
            stop_event.set()
            return


async def chat_ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024, heartbeat=25.0)
    await ws.prepare(request)

    service: QwenStreamingService = request.app["service"]
    llm_client: LMStudioStreamingClient = request.app["llm_client"]

    stop_event = threading.Event()
    mode = "custom_voice"          # resolved from payload below
    ref_audio_bytes: Optional[bytes] = None
    ref_text: str = ""

    llm_done = asyncio.Event()
    text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    full_text_parts: list[str] = []

    llm_task: Optional[asyncio.Task[None]] = None
    tts_task: Optional[asyncio.Task[None]] = None
    control_task: Optional[asyncio.Task[None]] = None

    try:
        first_msg = await ws.receive(timeout=30.0)
        if first_msg.type != WSMsgType.TEXT:
            await _safe_send_json(
                ws,
                {"type": "error", "message": "Expected JSON text frame as first message."},
            )
            return ws

        try:
            payload = json.loads(first_msg.data)
        except json.JSONDecodeError:
            await _safe_send_json(ws, {"type": "error", "message": "Invalid JSON payload."})
            return ws

        if not isinstance(payload, dict):
            await _safe_send_json(ws, {"type": "error", "message": "Payload must be an object."})
            return ws

        user_message = str(payload.get("user_message", "")).strip()
        if not user_message:
            await _safe_send_json(ws, {"type": "error", "message": "user_message is required."})
            return ws

        system_prompt = str(payload.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        language = str(payload.get("language") or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
        instruction = str(payload.get("instruction") or "").strip()
        speaker = str(payload.get("speaker") or DEFAULT_SPEAKER).strip() or DEFAULT_SPEAKER
        temperature = _coerce_float(payload.get("temperature"), default=0.7, low=0.0, high=2.0)
        max_tokens = _coerce_int(payload.get("max_tokens"), default=1024, low=64, high=4096)

        # Resolve TTS mode and reference audio for voice cloning.
        server_ref_bytes: Optional[bytes] = request.app.get("ref_audio_bytes")
        server_ref_text: str = str(request.app.get("ref_text") or "")
        default_mode = "voice_clone" if server_ref_bytes else "custom_voice"
        req_mode = str(payload.get("mode") or default_mode).strip()
        mode = req_mode if req_mode in ("custom_voice", "voice_clone") else default_mode

        if mode == "voice_clone":
            client_ref_b64 = payload.get("reference_audio_b64")
            client_ref_text = str(payload.get("reference_text") or "").strip()
            if client_ref_b64:
                try:
                    ref_audio_bytes = base64.b64decode(client_ref_b64)
                    ref_text = client_ref_text
                except Exception:
                    await _safe_send_json(ws, {"type": "error", "message": "Invalid reference_audio_b64 encoding."})
                    return ws
            elif server_ref_bytes:
                ref_audio_bytes = server_ref_bytes
                # If --ref-text was not supplied at startup, fall back to the
                # reference text the client sent (e.g. from the UI textarea).
                ref_text = server_ref_text or client_ref_text
            else:
                await _safe_send_json(ws, {"type": "error", "message": (
                    "voice_clone mode requires reference audio. "
                    "Start the server with --ref-audio, or send reference_audio_b64 in the request."
                )})
                return ws

        service.register_active_stream(mode, stop_event)

        model = await service.ensure_model(mode)
        if mode == "custom_voice":
            speaker = service.normalize_speaker(speaker, model)
        sample_rate = service.sampling_rate_for_mode(mode)

        await _safe_send_json(
            ws,
            {
                "type": "status",
                "message": "Session started",
                "sample_rate": sample_rate,
            },
        )
        await _safe_send_json(
            ws,
            {
                "type": "audio_meta",
                "sample_rate": sample_rate,
                "format": "pcm16le",
            },
        )

        async def llm_worker() -> None:
            buffer_text = ""
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
                async for delta in llm_client.stream_chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if stop_event.is_set():
                        break
                    full_text_parts.append(delta)
                    buffer_text += delta
                    if not await _safe_send_json(ws, {"type": "text_delta", "delta": delta}):
                        stop_event.set()
                        break
                    ready_segments, buffer_text = _extract_ready_segments(buffer_text)
                    for segment in ready_segments:
                        await text_queue.put(segment)
            except Exception as exc:
                logger.exception("LLM stream failed")
                await _safe_send_json(ws, {"type": "error", "message": f"LLM stream failed: {exc}"})
                stop_event.set()
            finally:
                tail = buffer_text.strip()
                if tail:
                    await text_queue.put(tail)
                await text_queue.put(None)
                llm_done.set()
                await _safe_send_json(ws, {"type": "llm_done"})

        async def tts_worker() -> None:
            chunk_index = 0
            segment_index = 0
            try:
                while not stop_event.is_set():
                    segment = await text_queue.get()
                    if segment is None:
                        break
                    segment_index += 1
                    params = StreamParams(
                        text=segment,
                        mode=mode,
                        language=language,
                        speaker=speaker,
                        instruction=instruction,
                        reference_audio_bytes=ref_audio_bytes,
                        reference_text=ref_text,
                    )
                    stream_iter = service.stream_pcm16(params=params, stop_event=stop_event)
                    async for pcm_chunk in stream_iter:
                        if stop_event.is_set():
                            break
                        if not pcm_chunk:
                            continue
                        chunk_index += 1
                        payload = {
                            "type": "audio_chunk",
                            "chunk_index": chunk_index,
                            "segment_index": segment_index,
                            "pcm16_b64": base64.b64encode(pcm_chunk).decode("ascii"),
                        }
                        if not await _safe_send_json(ws, payload):
                            stop_event.set()
                            break
                    if stop_event.is_set():
                        break
            except Exception as exc:
                logger.exception("Audio stream failed")
                await _safe_send_json(ws, {"type": "error", "message": f"Audio stream failed: {exc}"})
                stop_event.set()
            finally:
                if not stop_event.is_set():
                    await _safe_send_json(ws, {"type": "audio_done"})

        llm_task = asyncio.create_task(llm_worker(), name="llm_worker")
        tts_task = asyncio.create_task(tts_worker(), name="tts_worker")
        control_task = asyncio.create_task(
            _client_control_loop(ws=ws, stop_event=stop_event),
            name="ws_control_worker",
        )

        workers = asyncio.gather(llm_task, tts_task)
        done, pending = await asyncio.wait(
            {workers, control_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if control_task in done and not workers.done():
            stop_event.set()
            await asyncio.wait_for(workers, timeout=10.0)
        elif workers in done and not control_task.done():
            stop_event.set()
            control_task.cancel()
            try:
                await control_task
            except asyncio.CancelledError:
                pass

        for task in pending:
            task.cancel()

        await llm_done.wait()
        full_text = "".join(full_text_parts).strip()
        await _safe_send_json(ws, {"type": "done", "assistant_text": full_text})
        return ws
    except asyncio.TimeoutError:
        await _safe_send_json(
            ws,
            {"type": "error", "message": "Timed out waiting for initial request payload."},
        )
        return ws
    finally:
        stop_event.set()
        service.clear_active_stream(mode, stop_event)
        if llm_task is not None and not llm_task.done():
            llm_task.cancel()
        if tts_task is not None and not tts_task.done():
            tts_task.cancel()
        if control_task is not None and not control_task.done():
            control_task.cancel()
        if not ws.closed:
            await ws.close()


async def on_cleanup(app: web.Application) -> None:
    llm_client: LMStudioStreamingClient = app["llm_client"]
    await llm_client.close()


def build_app(
    custom_model_id: str,
    voice_clone_model_id: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    llm_api_name: str,
    llm_api_url: str,
    llm_api_key: str,
    llm_model_name: str,
    ref_audio_bytes: Optional[bytes] = None,
    ref_text: str = "",
    asr_url: str = DEFAULT_ASR_URL,
) -> web.Application:
    static_dir = Path(__file__).parent / "static_llm_chat"
    if not static_dir.exists():
        raise FileNotFoundError(f"Static directory not found: {static_dir}")

    app = web.Application(client_max_size=8 * 1024 * 1024)
    app["static_dir"] = str(static_dir)
    app["ref_audio_bytes"] = ref_audio_bytes
    app["ref_text"] = ref_text
    app["asr_url"] = asr_url
    app["service"] = QwenStreamingService(
        custom_model_id=custom_model_id,
        voice_clone_model_id=voice_clone_model_id,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    app["llm_client"] = LMStudioStreamingClient(
        api_name=llm_api_name,
        api_url=llm_api_url,
        api_key=llm_api_key,
        model_name=llm_model_name,
    )

    app.router.add_get("/", index_handler)
    app.router.add_get("/api/health", health_handler)
    app.router.add_get("/ws/chat", chat_ws_handler)
    app.router.add_post("/api/asr/transcribe", asr_transcribe_handler)
    app.router.add_static("/static-llm", path=str(static_dir))
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime LLM chat + live TTS audio streaming server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument(
        "--custom-model-id",
        default=os.environ.get(
            "QWEN_TTS_CUSTOM_MODEL_ID",
            os.environ.get("QWEN_TTS_MODEL_ID", DEFAULT_CUSTOM_MODEL_ID),
        ),
    )
    parser.add_argument(
        "--voice-clone-model-id",
        default=os.environ.get(
            "QWEN_TTS_VOICE_CLONE_MODEL_ID",
            DEFAULT_VOICE_CLONE_MODEL_ID,
        ),
    )
    parser.add_argument("--device", default=os.environ.get("QWEN_TTS_DEVICE", "auto"))
    parser.add_argument("--dtype", default=os.environ.get("QWEN_TTS_DTYPE", "bfloat16"))
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("QWEN_TTS_ATTN_IMPLEMENTATION", "sdpa"),
    )
    parser.add_argument(
        "--llm-api-name",
        default=os.environ.get("LLM_API_NAME", DEFAULT_LLM_API_NAME),
    )
    parser.add_argument(
        "--llm-api-url",
        default=os.environ.get("LLM_API_URL", DEFAULT_LLM_API_URL),
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("LLM_API_KEY", DEFAULT_LLM_API_KEY),
    )
    parser.add_argument(
        "--llm-model-name",
        default=os.environ.get("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME),
    )
    parser.add_argument(
        "--ref-audio",
        default=os.environ.get("TTS_REF_AUDIO", ""),
        help="Path to a WAV file used as the voice clone reference audio.",
    )
    parser.add_argument(
        "--ref-text",
        default=os.environ.get("TTS_REF_TEXT", ""),
        help="Transcript matching the reference audio (required for voice cloning).",
    )
    parser.add_argument(
        "--asr-url",
        default=os.environ.get("ASR_URL", DEFAULT_ASR_URL),
        help="URL of the Qwen3-ASR service used for voice input (default: %(default)s).",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        default=False,
        help="Serve plain HTTP instead of HTTPS (HTTPS is on by default for mic access).",
    )
    return parser.parse_args()


def ensure_ssl_cert() -> ssl.SSLContext:
    """Generate a self-signed cert (if absent) and return an SSLContext."""
    _SSL_DIR.mkdir(parents=True, exist_ok=True)
    if not (_CERT_PEM.exists() and _KEY_PEM.exists()):
        logger.info("Generating self-signed SSL certificate → %s", _SSL_DIR)
        cfg_path = _SSL_DIR / "openssl.cnf"
        cfg_path.write_text(
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
        subprocess.run(
            [
                "openssl", "req", "-x509",
                "-newkey", "rsa:2048",
                "-keyout", str(_KEY_PEM),
                "-out",    str(_CERT_PEM),
                "-days",   "3650",
                "-nodes",
                "-config", str(cfg_path),
            ],
            check=True,
            capture_output=True,
        )
        logger.info("SSL cert generated ✓")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(_CERT_PEM), str(_KEY_PEM))
    return ctx


def main() -> None:
    args = parse_args()

    ref_audio_bytes: Optional[bytes] = None
    ref_text: str = args.ref_text
    if args.ref_audio:
        ref_path = Path(args.ref_audio)
        if not ref_path.exists():
            logger.error("Reference audio file not found: %s", ref_path)
        else:
            ref_audio_bytes = ref_path.read_bytes()
            logger.info(
                "Loaded reference audio: %s (%d bytes)", ref_path, len(ref_audio_bytes)
            )

    app = build_app(
        custom_model_id=args.custom_model_id,
        voice_clone_model_id=args.voice_clone_model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        llm_api_name=args.llm_api_name,
        llm_api_url=args.llm_api_url,
        llm_api_key=args.llm_api_key,
        llm_model_name=args.llm_model_name,
        ref_audio_bytes=ref_audio_bytes,
        ref_text=ref_text,
        asr_url=args.asr_url,
    )
    ssl_ctx: Optional[ssl.SSLContext] = None
    if not args.no_ssl:
        ssl_ctx = ensure_ssl_cert()
        logger.info("HTTPS enabled — open https://%s:%d", args.host, args.port)
        logger.info("(Accept the self-signed cert warning once to enable mic access)")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
