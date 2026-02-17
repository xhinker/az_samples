#!/usr/bin/env python3
"""AIOHTTP realtime streaming TTS server for local MOSS-TTS-Realtime.

This script serves:
- GET /      : a minimal web UI
- GET /ws    : websocket endpoint to stream generated PCM chunks

WebSocket client payload:
{
  "text": "Text to speak",
  "user_text": "Optional user message context",
  "chunk_tokens": 6,
  "temperature": 0.8,
  "top_p": 0.6,
  "top_k": 30,
  "repetition_penalty": 1.1,
  "repetition_window": 50,
  "max_length": 3000
}
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime
import importlib.util
import json
import os
import sys
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from aiohttp import WSMsgType, web
from transformers import AutoModel, AutoTokenizer, __version__ as TRANSFORMERS_VERSION

# Disable the problematic cuDNN SDPA path; keep safe fallbacks.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tts_hf_models"
    / "OpenMOSS-Team"
    / "MOSS-TTS-Realtime_main"
)
DEFAULT_CODEC_PATH = (
    BASE_DIR
    / "models"
    / "tts_hf_models"
    / "OpenMOSS-Team"
    / "MOSS-Audio-Tokenizer_main"
)
DEFAULT_RT_PKG_PATH = BASE_DIR / "repos" / "MOSS-TTS" / "moss_tts_realtime"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_RATE = 24000


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _ensure_transformers_compat() -> None:
    min_version = (5, 0, 0)
    cur_version = _version_tuple(TRANSFORMERS_VERSION)

    if cur_version < min_version:
        raise RuntimeError(
            "MOSS-TTS-Realtime requires transformers>=5.0.0. "
            f"Found transformers=={TRANSFORMERS_VERSION}. "
            "Please upgrade in your runtime environment, for example:\n"
            "pip install -U 'transformers==5.0.0'"
        )

    if importlib.util.find_spec("transformers.models.qwen3") is None:
        raise RuntimeError(
            "Your transformers build does not provide 'transformers.models.qwen3'. "
            f"Current transformers version: {TRANSFORMERS_VERSION}. "
            "Install transformers==5.0.0 (or newer with Qwen3 support)."
        )

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MOSS-TTS Realtime Stream</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #222; }
    .panel { max-width: 840px; margin: 0 auto; }
    textarea { width: 100%; min-height: 180px; padding: 12px; box-sizing: border-box; font-size: 15px; }
    .row { display: flex; gap: 10px; margin-top: 12px; }
    button { padding: 10px 14px; font-size: 14px; cursor: pointer; }
    input[type=number], input[type=text] { padding: 8px; width: 140px; }
    .status { margin-top: 12px; font-size: 13px; color: #444; white-space: pre-wrap; }
    .muted { color: #666; font-size: 12px; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="panel">
    <h2>MOSS-TTS Realtime (AIOHTTP)</h2>
    <p class="muted">Enter text and click Stream. Audio plays progressively as chunks arrive.</p>

    <textarea id="text" placeholder="Type text to synthesize...">Welcome to MOSS TTS Realtime. This demo streams generated speech chunks to the browser in real time.</textarea>

    <div class="row">
      <input id="chunkTokens" type="number" min="1" value="6" title="text token chunk size" />
      <button id="startBtn">Stream</button>
      <button id="stopBtn">Stop</button>
    </div>

    <div id="status" class="status">Idle.</div>
  </div>

<script>
(() => {
  const textEl = document.getElementById('text');
  const chunkTokensEl = document.getElementById('chunkTokens');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const statusEl = document.getElementById('status');

let ws = null;
let audioCtx = null;
let nextTime = 0;
let streamPrimed = false;

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function b64ToFloat32(base64) {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  }

  async function ensureAudio(sampleRate) {
    if (!audioCtx || audioCtx.sampleRate !== sampleRate) {
      if (audioCtx) {
        await audioCtx.close();
      }
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
      nextTime = audioCtx.currentTime;
    }
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }
  }

async function playChunk(samples, sampleRate) {
  await ensureAudio(sampleRate);

  const buffer = audioCtx.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  if (!streamPrimed) {
    // One-time startup cushion to absorb initial websocket jitter.
    nextTime = now + 0.08;
    streamPrimed = true;
  } else if (nextTime < now) {
    // If underrun happens, catch up immediately without inserting extra silence.
    nextTime = now;
  }

  const startAt = nextTime;
  source.start(startAt);
  nextTime = startAt + buffer.duration;
}

  function closeSocket() {
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  startBtn.addEventListener('click', async () => {
    const text = textEl.value.trim();
    if (!text) {
      setStatus('Please provide text.');
      return;
    }

    closeSocket();
    streamPrimed = false;
    nextTime = 0;
    setStatus('Connecting...');

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => {
      setStatus('Streaming...');
      const payload = {
        text,
        chunk_tokens: Number(chunkTokensEl.value || 6),
      };
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'audio_chunk') {
          const samples = b64ToFloat32(msg.pcm_f32);
          await playChunk(samples, msg.sample_rate);
          setStatus(`Streaming chunk #${msg.index}`);
          return;
        }
        if (msg.type === 'done') {
          if (msg.saved_wav) {
            setStatus(`Done. Saved: ${msg.saved_wav}`);
          } else {
            setStatus('Done.');
          }
          return;
        }
        if (msg.type === 'error') {
          setStatus(`Error: ${msg.message}`);
          return;
        }
        if (msg.type === 'info') {
          setStatus(msg.message);
        }
      } catch (err) {
        setStatus(`Client parse error: ${String(err)}`);
      }
    };

    ws.onerror = () => {
      setStatus('WebSocket error.');
    };

    ws.onclose = () => {
      ws = null;
    };
  });

  stopBtn.addEventListener('click', () => {
    closeSocket();
    streamPrimed = false;
    nextTime = 0;
    setStatus('Stopped.');
  });
})();
</script>
</body>
</html>
"""


def _resolve_attn_implementation(device: torch.device, dtype: torch.dtype) -> str:
    # Prefer SDPA for runtime stability in streaming mode.
    if device.type == "cuda":
        return "sdpa"
    return "eager"


def _sanitize_tokens(
    tokens: torch.Tensor,
    codebook_size: int,
    audio_eos_token: int,
) -> tuple[torch.Tensor, bool]:
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens, False

    eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False)
    invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)

    stop_idx = None
    if eos_rows.numel() > 0:
        stop_idx = int(eos_rows[0].item())
    if invalid_rows.any():
        invalid_idx = int(invalid_rows.nonzero(as_tuple=False)[0].item())
        stop_idx = invalid_idx if stop_idx is None else min(stop_idx, invalid_idx)

    if stop_idx is not None:
        return tokens[:stop_idx], True
    return tokens, False


def _build_text_only_turn_input(processor, user_text: str, prompt_tokens: np.ndarray | None = None) -> np.ndarray:
    system_prompt = processor.make_ensemble(prompt_tokens)
    user_prompt_text = "<|im_end|>\n<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n"
    user_prompt_tokens = processor.tokenizer(user_prompt_text)["input_ids"]
    user_prompt = np.full(
        shape=(len(user_prompt_tokens), processor.channels + 1),
        fill_value=processor.audio_channel_pad,
        dtype=np.int64,
    )
    user_prompt[:, 0] = np.asarray(user_prompt_tokens, dtype=np.int64)
    return np.concatenate([system_prompt, user_prompt], axis=0)


class RealtimeTTSRuntime:
    def __init__(self, model_path: Path, codec_path: Path, rt_pkg_path: Path):
        self.model_path = model_path
        self.codec_path = codec_path
        self.rt_pkg_path = rt_pkg_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.attn_impl = _resolve_attn_implementation(self.device, self.dtype)

        self._ready = False
        self._load_lock = asyncio.Lock()
        self._stream_lock = asyncio.Lock()

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.codec = None

        self.MossTTSRealtime = None
        self.MossTTSRealtimeProcessor = None
        self.MossTTSRealtimeInference = None
        self.MossTTSRealtimeStreamingSession = None
        self.AudioStreamDecoder = None

    def _new_output_wav_path(self) -> Path:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return OUTPUTS_DIR / f"stream_{stamp}.wav"

    async def ensure_loaded(self) -> None:
        if self._ready:
            return

        async with self._load_lock:
            if self._ready:
                return

            if not self.rt_pkg_path.exists():
                raise FileNotFoundError(f"Realtime package path not found: {self.rt_pkg_path}")
            if str(self.rt_pkg_path) not in sys.path:
                sys.path.insert(0, str(self.rt_pkg_path))

            _ensure_transformers_compat()

            from mossttsrealtime import MossTTSRealtime, MossTTSRealtimeProcessor
            from mossttsrealtime.streaming_mossttsrealtime import (
                AudioStreamDecoder,
                MossTTSRealtimeInference,
                MossTTSRealtimeStreamingSession,
            )

            self.MossTTSRealtime = MossTTSRealtime
            self.MossTTSRealtimeProcessor = MossTTSRealtimeProcessor
            self.MossTTSRealtimeInference = MossTTSRealtimeInference
            self.MossTTSRealtimeStreamingSession = MossTTSRealtimeStreamingSession
            self.AudioStreamDecoder = AudioStreamDecoder

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self.processor = self.MossTTSRealtimeProcessor(self.tokenizer)

            self.model = self.MossTTSRealtime.from_pretrained(
                str(self.model_path),
                attn_implementation=self.attn_impl,
                torch_dtype=self.dtype,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()

            self.codec = AutoModel.from_pretrained(
                str(self.codec_path),
                trust_remote_code=True,
                local_files_only=True,
            ).eval().to(self.device)

            self._ready = True

    def _decode_audio_frames(
        self,
        audio_frames: list[torch.Tensor],
        decoder,
        codebook_size: int,
        audio_eos_token: int,
    ) -> Iterator[torch.Tensor]:
        for frame in audio_frames:
            tokens = frame
            if tokens.dim() == 3:
                tokens = tokens[0]
            if tokens.dim() != 2:
                raise ValueError(f"Expected [T, C] audio tokens, got {tuple(tokens.shape)}")

            tokens, stop = _sanitize_tokens(tokens, codebook_size, audio_eos_token)
            if tokens.numel() == 0:
                if stop:
                    break
                continue

            decoder.push_tokens(tokens.detach())
            for wav in decoder.audio_chunks():
                if wav.numel() == 0:
                    continue
                yield wav.detach().cpu()

            if stop:
                break

    async def stream_text_to_ws(
        self,
        ws: web.WebSocketResponse,
        text: str,
        *,
        user_text: str,
        chunk_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        repetition_window: int,
        max_length: int,
    ) -> None:
        await self.ensure_loaded()

        async with self._stream_lock:
            inferencer = self.MossTTSRealtimeInference(self.model, self.tokenizer, max_length=max_length)
            inferencer.reset_generation_state(keep_cache=False)

            session = self.MossTTSRealtimeStreamingSession(
                inferencer,
                self.processor,
                codec=self.codec,
                codec_sample_rate=SAMPLE_RATE,
                codec_encode_kwargs={"chunk_duration": 0.24},
                prefill_text_len=self.processor.delay_tokens_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
            )

            turn_input_ids = _build_text_only_turn_input(self.processor, user_text=user_text, prompt_tokens=None)
            session.reset_turn(input_ids=turn_input_ids, include_system_prompt=True, reset_cache=True)

            decoder = self.AudioStreamDecoder(
                self.codec,
                chunk_frames=3,
                overlap_frames=0,
                decode_kwargs={"chunk_duration": -1},
                device=self.device,
            )

            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if not text_tokens:
                raise ValueError("Tokenization produced no tokens.")

            step = max(1, int(chunk_tokens))
            codebook_size = int(getattr(self.codec, "codebook_size", 1024))
            audio_eos_token = int(getattr(inferencer, "audio_eos_token", 1026))

            chunk_index = 0
            out_wav_path = self._new_output_wav_path()

            with wave.open(str(out_wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)

                async def _send_chunk(wav_chunk: torch.Tensor) -> None:
                    nonlocal chunk_index
                    audio_f32 = wav_chunk.to(torch.float32).contiguous().numpy().reshape(-1)
                    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
                    pcm16 = (audio_f32 * 32767.0).round().astype(np.int16)
                    wf.writeframes(pcm16.tobytes())

                    chunk_index += 1
                    payload = base64.b64encode(audio_f32.tobytes()).decode("ascii")
                    await ws.send_json(
                        {
                            "type": "audio_chunk",
                            "index": chunk_index,
                            "sample_rate": SAMPLE_RATE,
                            "pcm_f32": payload,
                        }
                    )

                with self.codec.streaming(batch_size=1):
                    for start in range(0, len(text_tokens), step):
                        token_chunk = text_tokens[start : start + step]
                        audio_frames = session.push_text_tokens(token_chunk)
                        for wav_chunk in self._decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token):
                            await _send_chunk(wav_chunk)
                        await asyncio.sleep(0)

                    audio_frames = session.end_text()
                    for wav_chunk in self._decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token):
                        await _send_chunk(wav_chunk)

                    while True:
                        audio_frames = session.drain(max_steps=1)
                        if not audio_frames:
                            break
                        for wav_chunk in self._decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token):
                            await _send_chunk(wav_chunk)
                        if session.inferencer.is_finished:
                            break
                        await asyncio.sleep(0)

                    final_chunk = decoder.flush()
                    if final_chunk is not None and final_chunk.numel() > 0:
                        await _send_chunk(final_chunk.detach().cpu())

            try:
                saved_wav = str(out_wav_path.relative_to(BASE_DIR))
            except ValueError:
                saved_wav = str(out_wav_path)

            await ws.send_json({"type": "done", "chunks": chunk_index, "saved_wav": saved_wav})


async def index_handler(_: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def health_handler(request: web.Request) -> web.Response:
    runtime: RealtimeTTSRuntime = request.app["runtime"]
    return web.json_response(
        {
            "ok": True,
            "ready": runtime._ready,
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "model_path": str(runtime.model_path),
            "codec_path": str(runtime.codec_path),
            "attn_implementation": runtime.attn_impl,
        }
    )


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    runtime: RealtimeTTSRuntime = request.app["runtime"]
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)

    await ws.send_json({"type": "info", "message": "Connected. Send JSON with field 'text'."})

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue

            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON payload."})
                continue

            text = str(payload.get("text", "")).strip()
            if not text:
                await ws.send_json({"type": "error", "message": "Field 'text' is required."})
                continue

            user_text = str(payload.get("user_text") or "Please read the following text naturally.")
            chunk_tokens = int(payload.get("chunk_tokens", 6))
            temperature = float(payload.get("temperature", 0.8))
            top_p = float(payload.get("top_p", 0.6))
            top_k = int(payload.get("top_k", 30))
            repetition_penalty = float(payload.get("repetition_penalty", 1.1))
            repetition_window = int(payload.get("repetition_window", 50))
            max_length = int(payload.get("max_length", 3000))

            try:
                await runtime.stream_text_to_ws(
                    ws,
                    text,
                    user_text=user_text,
                    chunk_tokens=chunk_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                    max_length=max_length,
                )
            except Exception as exc:
                await ws.send_json({"type": "error", "message": str(exc)})

    finally:
        await ws.close()

    return ws


def create_app(runtime: RealtimeTTSRuntime) -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024)
    app["runtime"] = runtime
    app.router.add_get("/", index_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIOHTTP MOSS-TTS-Realtime streaming server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--codec-path", type=Path, default=DEFAULT_CODEC_PATH)
    parser.add_argument("--rt-pkg-path", type=Path, default=DEFAULT_RT_PKG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model_path.expanduser().resolve()
    codec_path = args.codec_path.expanduser().resolve()
    rt_pkg_path = args.rt_pkg_path.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not codec_path.exists():
        raise FileNotFoundError(f"Codec path not found: {codec_path}")
    if not rt_pkg_path.exists():
        raise FileNotFoundError(f"Realtime package path not found: {rt_pkg_path}")

    runtime = RealtimeTTSRuntime(
        model_path=model_path,
        codec_path=codec_path,
        rt_pkg_path=rt_pkg_path,
    )
    app = create_app(runtime)

    print(f"[INFO] host={args.host} port={args.port}")
    print(f"[INFO] model={model_path}")
    print(f"[INFO] codec={codec_path}")
    print(f"[INFO] device={runtime.device} dtype={runtime.dtype} attn={runtime.attn_impl}")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
