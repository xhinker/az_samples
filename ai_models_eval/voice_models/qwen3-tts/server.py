#!/usr/bin/env python3
"""Aiohttp server for Qwen3-TTS streaming (PCM16 over HTTP chunked response)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import tempfile
from queue import Empty, Queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
STREAM_DECODE_EVERY_FRAMES = 6
STREAM_POLL_SECONDS = 0.01


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


@dataclass
class PreparedStreamRequest:
    generate_kwargs: dict[str, Any]
    ref_code: Optional[torch.Tensor]


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
        # True streaming hooks into model internals; keep one active generation per model.
        self._stream_locks: dict[str, asyncio.Lock] = {
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

    def _build_stream_request_blocking(
        self,
        model,
        params: StreamParams,
    ) -> PreparedStreamRequest:
        text = params.text.strip()
        if not text:
            raise ValueError("Text is required.")

        input_ids = model._tokenize_texts([model._build_assistant_text(text)])
        languages = [params.language or DEFAULT_LANGUAGE]
        model._validate_languages(languages)
        gen_kwargs = model._merge_generate_kwargs(non_streaming_mode=False)

        if params.mode == "custom_voice":
            speakers = [params.speaker]
            model._validate_speakers(speakers)

            instruct_ids = [None]
            if params.instruction:
                instruct_ids = [model._tokenize_texts([model._build_instruct_text(params.instruction)])[0]]

            return PreparedStreamRequest(
                generate_kwargs={
                    "input_ids": input_ids,
                    "instruct_ids": instruct_ids,
                    "languages": languages,
                    "speakers": speakers,
                    "non_streaming_mode": False,
                    **gen_kwargs,
                },
                ref_code=None,
            )

        if params.mode != "voice_clone":
            raise ValueError(f"Unsupported mode: {params.mode!r}")
        if not params.reference_audio_bytes:
            raise ValueError("Reference audio file is required in voice_clone mode.")
        if not params.reference_text.strip():
            raise ValueError("Reference transcript is required in voice_clone mode.")

        prompt_items = self._create_voice_clone_prompt(
            model,
            params.reference_audio_bytes,
            params.reference_text,
        )
        if len(prompt_items) != 1:
            raise ValueError(f"Expected 1 prompt item, got {len(prompt_items)}.")

        voice_clone_prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
        ref_texts_for_ids = [prompt_items[0].ref_text]
        ref_ids = []
        for ref_text in ref_texts_for_ids:
            if ref_text is None or ref_text == "":
                ref_ids.append(None)
            else:
                ref_ids.append(model._tokenize_texts([model._build_ref_text(ref_text)])[0])

        ref_code = None
        ref_code_list = voice_clone_prompt.get("ref_code")
        if ref_code_list and ref_code_list[0] is not None:
            ref_code = ref_code_list[0].detach().to("cpu", dtype=torch.long)

        return PreparedStreamRequest(
            generate_kwargs={
                "input_ids": input_ids,
                "ref_ids": ref_ids,
                "voice_clone_prompt": voice_clone_prompt,
                "languages": languages,
                "non_streaming_mode": False,
                **gen_kwargs,
            },
            ref_code=ref_code,
        )

    def _run_generate_with_hook_blocking(
        self,
        model,
        prepared: PreparedStreamRequest,
        events: Queue,
    ) -> None:
        talker = model.model.talker

        def on_forward(_module, _inputs, out):
            try:
                hidden_states = getattr(out, "hidden_states", None)
                if isinstance(hidden_states, (tuple, list)) and hidden_states:
                    codec_ids = hidden_states[-1]
                    if codec_ids is not None:
                        frame = codec_ids[0].detach().to("cpu", dtype=torch.long)
                        events.put(("frame", frame))
            except Exception as hook_exc:  # pragma: no cover
                events.put(("error", f"stream hook failed: {hook_exc}"))

        hook_handle = talker.register_forward_hook(on_forward)
        try:
            model.model.generate(**prepared.generate_kwargs)
        except Exception as exc:
            events.put(("error", str(exc)))
        finally:
            hook_handle.remove()
            events.put(("done", None))

    def _decode_codes_to_audio_blocking(
        self,
        model,
        generated_codes: torch.Tensor,
        ref_code: Optional[torch.Tensor],
    ) -> tuple[np.ndarray, int]:
        if generated_codes.numel() == 0:
            return np.zeros(0, dtype=np.float32), self.sampling_rate_for_mode("voice_clone")

        decode_codes = generated_codes
        ref_len = 0
        if ref_code is not None and ref_code.numel() > 0:
            ref_len = int(ref_code.shape[0])
            decode_codes = torch.cat([ref_code, generated_codes], dim=0)

        wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": decode_codes}])
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)

        if ref_len > 0:
            total_len = int(decode_codes.shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]

        return wav, int(sample_rate)

    async def stream_pcm16(self, params: StreamParams):
        model = await self.ensure_model(params.mode)
        expected_sr = self.sampling_rate_for_mode(params.mode)
        loop = asyncio.get_running_loop()
        prepared = await loop.run_in_executor(
            None,
            self._build_stream_request_blocking,
            model,
            params,
        )

        stream_lock = self._stream_locks[params.mode]
        async with stream_lock:
            events: Queue = Queue()
            generation_future = loop.run_in_executor(
                None,
                self._run_generate_with_hook_blocking,
                model,
                prepared,
                events,
            )

            emitted_samples = 0
            decoded_frames = 0
            frame_buffer: list[torch.Tensor] = []
            done = False

            while True:
                had_event = False
                while True:
                    try:
                        event, payload = events.get_nowait()
                    except Empty:
                        break
                    had_event = True
                    if event == "frame":
                        frame_buffer.append(payload)
                    elif event == "error":
                        raise RuntimeError(payload)
                    elif event == "done":
                        done = True

                frames_ready = len(frame_buffer) - decoded_frames
                should_decode = (
                    frames_ready >= STREAM_DECODE_EVERY_FRAMES
                    or (done and frames_ready > 0)
                )

                if should_decode:
                    codes = torch.stack(frame_buffer, dim=0)
                    wav_f32, sample_rate = await loop.run_in_executor(
                        None,
                        self._decode_codes_to_audio_blocking,
                        model,
                        codes,
                        prepared.ref_code,
                    )
                    decoded_frames = len(frame_buffer)

                    if sample_rate != expected_sr:
                        logger.warning(
                            "Decoded sample rate (%s) differs from expected sample rate (%s).",
                            sample_rate,
                            expected_sr,
                        )

                    if wav_f32.size > emitted_samples:
                        delta = wav_f32[emitted_samples:]
                        emitted_samples = wav_f32.size
                        pcm16 = np.clip(delta * 32767.0, -32768, 32767).astype(np.int16)
                        if pcm16.size > 0:
                            yield pcm16.tobytes()

                if done and decoded_frames >= len(frame_buffer):
                    break
                if not had_event:
                    await asyncio.sleep(STREAM_POLL_SECONDS)

            await generation_future


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
