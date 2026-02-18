#!/usr/bin/env python3
"""Aiohttp server for Qwen3-TTS streaming (PCM16 over HTTP chunked response)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from aiohttp import web
from aiohttp.web_request import FileField

try:
    import torch
    from qwen_tts import Qwen3TTSModel
except ImportError as exc:
    torch = None
    Qwen3TTSModel = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("qwen3_tts_server")

DEFAULT_CUSTOM_MODEL_ID = (
    "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice_main"
)
DEFAULT_VOICE_CLONE_MODEL_ID = (
    "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main"
)
DEFAULT_SPEAKER = "vivian"
DEFAULT_LANGUAGE = "Auto"
WORDS_PER_CHUNK = 12
SILENCE_TRIM_THRESHOLD = 1e-3
KEEP_TAIL_SILENCE_SECONDS = 0.08


def split_text_for_streaming(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    clean = " ".join(text.strip().split())
    if not clean:
        return []

    # Split on sentence and major clause punctuation to reduce long per-chunk latency.
    sentences = [s for s in re.split(r"(?<=[.!?。！？,，;；:：])\s+", clean) if s]
    chunks: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        while len(words) > words_per_chunk:
            chunks.append(" ".join(words[:words_per_chunk]))
            words = words[words_per_chunk:]
        if words:
            chunks.append(" ".join(words))
    return chunks


def trim_trailing_silence(
    wav_f32: np.ndarray,
    sample_rate: int,
    threshold: float = SILENCE_TRIM_THRESHOLD,
    keep_tail_seconds: float = KEEP_TAIL_SILENCE_SECONDS,
) -> np.ndarray:
    if wav_f32.size == 0:
        return wav_f32

    non_silent = np.flatnonzero(np.abs(wav_f32) > threshold)
    if non_silent.size == 0:
        return wav_f32

    last_non_silent = int(non_silent[-1])
    keep_tail_samples = max(1, int(sample_rate * keep_tail_seconds))
    cut_index = min(wav_f32.size, last_non_silent + 1 + keep_tail_samples)
    return wav_f32[:cut_index]


def _resolve_dtype(dtype_str: str):
    if torch is None:
        return None
    key = dtype_str.lower().strip()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(key, torch.bfloat16)


@dataclass
class StreamParams:
    text: str
    mode: str
    language: str
    speaker: str
    instruction: str
    reference_audio_bytes: Optional[bytes]
    reference_text: str


class QwenStreamingService:
    def __init__(
        self,
        custom_model_id: str,
        voice_clone_model_id: str,
        device: str,
        dtype: str,
        attn_implementation: str,
    ):
        self.custom_model_id = custom_model_id
        self.voice_clone_model_id = voice_clone_model_id
        self.device = device
        self.dtype = _resolve_dtype(dtype)
        self.attn_implementation = attn_implementation
        self._models: dict[str, object] = {}
        self._sampling_rates: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {
            "custom_voice": asyncio.Lock(),
            "voice_clone": asyncio.Lock(),
        }

    @property
    def model_ids(self) -> dict[str, str]:
        return {
            "custom_voice": self.custom_model_id,
            "voice_clone": self.voice_clone_model_id,
        }

    def sampling_rate_for_mode(self, mode: str) -> int:
        return int(self._sampling_rates.get(mode, 24000))

    def get_supported_speakers(self, model) -> Optional[list[str]]:
        if model is None:
            return None
        getter = getattr(model, "get_supported_speakers", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                logger.exception("Failed to query supported speakers.")
        return None

    def normalize_speaker(self, speaker: str, model) -> str:
        supported = self.get_supported_speakers(model)
        if not supported:
            return speaker

        normalized = (speaker or "").strip().lower()
        if not normalized:
            return supported[0]

        supported_set = {str(s).lower() for s in supported}
        if normalized not in supported_set:
            raise ValueError(
                f"Unsupported speaker: {speaker!r}. Supported: {sorted(supported_set)}"
            )
        return normalized

    def _model_id_for_mode(self, mode: str) -> str:
        if mode == "custom_voice":
            return self.custom_model_id
        if mode == "voice_clone":
            return self.voice_clone_model_id
        raise ValueError(f"Unsupported mode: {mode!r}")

    async def ensure_model(self, mode: str):
        if mode in self._models:
            return self._models[mode]

        lock = self._locks[mode]
        async with lock:
            if mode in self._models:
                return self._models[mode]

            if IMPORT_ERROR is not None:
                raise RuntimeError(
                    "qwen_tts import failed. Install dependencies first."
                ) from IMPORT_ERROR

            loop = asyncio.get_running_loop()
            model_id = self._model_id_for_mode(mode)
            model = await loop.run_in_executor(None, self._load_model_blocking, model_id)
            sample_rate = int(getattr(model, "sampling_rate", 24000))
            self._models[mode] = model
            self._sampling_rates[mode] = sample_rate
            logger.info(
                "Loaded %s model: %s (sampling_rate=%s)",
                mode,
                model_id,
                sample_rate,
            )
            return model

    def _load_model_blocking(self, model_id: str):
        kwargs = {
            "device_map": self.device,
        }
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
            kwargs["torch_dtype"] = self.dtype
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        try:
            return Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        except TypeError as exc:
            if "torch_dtype" in str(exc):
                logger.warning("Model loader rejected torch_dtype, retrying without it (%s)", exc)
                kwargs.pop("torch_dtype", None)
                return Qwen3TTSModel.from_pretrained(model_id, **kwargs)
            raise
        except Exception as exc:
            if "attn_implementation" in kwargs:
                logger.warning(
                    "Model load failed with attn_implementation=%s, retrying without it (%s)",
                    kwargs["attn_implementation"],
                    exc,
                )
                kwargs.pop("attn_implementation", None)
                return Qwen3TTSModel.from_pretrained(model_id, **kwargs)
            raise

    def _create_voice_clone_prompt(self, model, audio_bytes: bytes, reference_text: str):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(audio_bytes)
            temp_path = temp_wav.name
        try:
            return model.create_voice_clone_prompt(
                ref_audio=temp_path,
                ref_text=reference_text,
                x_vector_only_mode=False,
            )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning("Failed to delete temp file: %s", temp_path)

    def _generate_audio_chunk_blocking(
        self,
        model,
        text_chunk: str,
        mode: str,
        language: str,
        speaker: str,
        instruction: str,
        voice_clone_prompt: Optional[str],
    ):
        if mode == "voice_clone":
            if not voice_clone_prompt:
                raise ValueError("Voice clone mode requires a prepared voice clone prompt.")
            clone_kwargs = {
                "text": text_chunk,
                "language": language,
                "voice_clone_prompt": voice_clone_prompt,
                "non_streaming_mode": False,
            }
            if instruction:
                clone_kwargs["instruct"] = instruction
            try:
                wavs, sample_rate = model.generate_voice_clone(**clone_kwargs)
            except TypeError:
                clone_kwargs.pop("instruct", None)
                wavs, sample_rate = model.generate_voice_clone(**clone_kwargs)
        else:
            kwargs = {
                "text": text_chunk,
                "language": language,
                "speaker": speaker,
            }
            if instruction:
                kwargs["instruct"] = instruction
            wavs, sample_rate = model.generate_custom_voice(**kwargs)

        if isinstance(wavs, (list, tuple)):
            wav = wavs[0]
        else:
            wav = wavs
        wav_f32 = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav_f32, int(sample_rate)

    async def stream_pcm16(self, params: StreamParams):
        model = await self.ensure_model(params.mode)
        expected_sr = self.sampling_rate_for_mode(params.mode)

        chunks = split_text_for_streaming(params.text)
        if not chunks:
            raise ValueError("Text is empty after normalization.")

        voice_clone_prompt = None
        if params.mode == "voice_clone":
            if not params.reference_audio_bytes:
                raise ValueError("Reference audio file is required in voice_clone mode.")
            if not params.reference_text.strip():
                raise ValueError("Reference transcript is required in voice_clone mode.")
            loop = asyncio.get_running_loop()
            voice_clone_prompt = await loop.run_in_executor(
                None,
                self._create_voice_clone_prompt,
                model,
                params.reference_audio_bytes,
                params.reference_text,
            )

        total_chunks = len(chunks)
        for index, text_chunk in enumerate(chunks, start=1):
            t0 = time.perf_counter()
            loop = asyncio.get_running_loop()
            wav_f32, sample_rate = await loop.run_in_executor(
                None,
                self._generate_audio_chunk_blocking,
                model,
                text_chunk,
                params.mode,
                params.language,
                params.speaker,
                params.instruction,
                voice_clone_prompt,
            )
            gen_time = time.perf_counter() - t0

            if sample_rate != expected_sr:
                logger.warning(
                    "Chunk sample rate (%s) differs from model sample rate (%s).",
                    sample_rate,
                    expected_sr,
                )

            if wav_f32.size == 0:
                continue

            if index < total_chunks:
                wav_f32 = trim_trailing_silence(wav_f32, sample_rate)

            pcm16 = np.clip(wav_f32 * 32767.0, -32768, 32767).astype(np.int16)
            logger.info(
                "Chunk %s/%s generated in %.2fs, %.2fs audio",
                index,
                len(chunks),
                gen_time,
                wav_f32.size / float(sample_rate),
            )
            yield pcm16.tobytes()


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


async def stream_handler(request: web.Request):
    service: QwenStreamingService = request.app["service"]
    response: Optional[web.StreamResponse] = None

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

        stream_iter = service.stream_pcm16(params)
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
                break

        try:
            await response.write_eof()
        except ConnectionResetError:
            logger.info("Client disconnected before EOF.")
        return response

    except StopAsyncIteration:
        return web.json_response({"error": "Model generated empty audio output."}, status=400)
    except Exception as exc:
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
