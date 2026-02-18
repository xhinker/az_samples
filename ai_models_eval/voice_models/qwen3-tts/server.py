#!/usr/bin/env python3
"""Aiohttp server for Qwen3-TTS streaming (PCM16 over HTTP chunked response)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import tempfile
from queue import Empty, Queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np
from aiohttp import web
from aiohttp.web_request import FileField
from transformers import StoppingCriteria, StoppingCriteriaList

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
STREAM_LOOKBACK_FRAMES = 48
STREAM_MAX_DECODE_FRAMES = 384
DEFAULT_DECODE_HOP = 1920


class StopEventCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


class GenerationCancelled(Exception):
    pass


def estimate_max_new_tokens_from_text(text: str) -> int:
    words = len(text.split())
    chars = len(text)
    estimated = max(int(words * 7.0), int(chars * 1.6)) + 160
    return max(192, min(4096, estimated))


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
        self._decode_hops: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {
            "custom_voice": asyncio.Lock(),
            "voice_clone": asyncio.Lock(),
        }
        # True streaming hooks into model internals; keep one active generation per model.
        self._stream_locks: dict[str, asyncio.Lock] = {
            "custom_voice": asyncio.Lock(),
            "voice_clone": asyncio.Lock(),
        }
        self._active_stop_events: dict[str, threading.Event] = {}
        self._active_stop_events_lock = threading.Lock()

    @property
    def model_ids(self) -> dict[str, str]:
        return {
            "custom_voice": self.custom_model_id,
            "voice_clone": self.voice_clone_model_id,
        }

    def sampling_rate_for_mode(self, mode: str) -> int:
        return int(self._sampling_rates.get(mode, 24000))

    def decode_hop_for_mode(self, mode: str) -> int:
        return int(self._decode_hops.get(mode, DEFAULT_DECODE_HOP))

    def register_active_stream(self, mode: str, stop_event: threading.Event) -> None:
        with self._active_stop_events_lock:
            self._active_stop_events[mode] = stop_event

    def clear_active_stream(self, mode: str, stop_event: threading.Event) -> None:
        with self._active_stop_events_lock:
            current = self._active_stop_events.get(mode)
            if current is stop_event:
                self._active_stop_events.pop(mode, None)

    def request_stop(self, mode: Optional[str] = None) -> int:
        with self._active_stop_events_lock:
            if mode:
                target = self._active_stop_events.get(mode)
                if target is None:
                    return 0
                target.set()
                logger.info("Stop requested for mode=%s", mode)
                return 1

            count = 0
            for stop_event in self._active_stop_events.values():
                stop_event.set()
                count += 1
            if count > 0:
                logger.info("Stop requested for all active streams (count=%s)", count)
            return count

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
            decode_hop = DEFAULT_DECODE_HOP
            try:
                speech_tokenizer = model.model.speech_tokenizer
                hop_getter = getattr(speech_tokenizer, "get_decode_upsample_rate", None)
                if callable(hop_getter):
                    decode_hop = int(hop_getter())
            except Exception:
                logger.warning("Could not read decode hop from speech tokenizer, using default %s.", DEFAULT_DECODE_HOP)

            self._models[mode] = model
            self._sampling_rates[mode] = sample_rate
            self._decode_hops[mode] = max(1, decode_hop)
            logger.info(
                "Loaded %s model: %s (sampling_rate=%s, decode_hop=%s)",
                mode,
                model_id,
                sample_rate,
                self._decode_hops[mode],
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
        try:
            requested_max_new_tokens = int(gen_kwargs.get("max_new_tokens", 2048))
        except (TypeError, ValueError):
            requested_max_new_tokens = 2048
        safe_max_new_tokens = int(estimate_max_new_tokens_from_text(text) * 1.4)
        gen_kwargs["max_new_tokens"] = max(64, min(requested_max_new_tokens, safe_max_new_tokens))
        if gen_kwargs["max_new_tokens"] != requested_max_new_tokens:
            logger.info(
                "Adjusted max_new_tokens from %s to %s for text length=%s",
                requested_max_new_tokens,
                gen_kwargs["max_new_tokens"],
                len(text),
            )

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

    def _attach_stop_criteria(
        self,
        prepared: PreparedStreamRequest,
        stop_event: threading.Event,
    ) -> None:
        cancel_criteria = StopEventCriteria(stop_event)
        existing = prepared.generate_kwargs.get("stopping_criteria")
        if existing is None:
            prepared.generate_kwargs["stopping_criteria"] = StoppingCriteriaList([cancel_criteria])
            return

        if isinstance(existing, StoppingCriteriaList):
            prepared.generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                list(existing) + [cancel_criteria]
            )
            return

        if isinstance(existing, list):
            prepared.generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                existing + [cancel_criteria]
            )
            return

        prepared.generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [existing, cancel_criteria]
        )

    def _run_generate_with_hook_blocking(
        self,
        model,
        prepared: PreparedStreamRequest,
        events: Queue,
        stop_event: threading.Event,
    ) -> None:
        talker = model.model.talker
        eos_token_id = int(model.model.config.talker_config.codec_eos_token_id)

        def on_forward_pre(_module, _inputs):
            if stop_event.is_set():
                raise GenerationCancelled("generation cancelled by user")

        def on_forward(_module, _inputs, out):
            try:
                hidden_states = getattr(out, "hidden_states", None)
                if isinstance(hidden_states, (tuple, list)) and hidden_states:
                    codec_ids = hidden_states[-1]
                    if codec_ids is not None:
                        frame = codec_ids[0].detach().to("cpu", dtype=torch.long).view(-1)
                        if frame.numel() == 0:
                            return
                        if int(frame[0].item()) == eos_token_id:
                            stop_event.set()
                            events.put(("eos", None))
                            return
                        events.put(("frame", frame))
            except Exception as hook_exc:  # pragma: no cover
                events.put(("error", f"stream hook failed: {hook_exc}"))

        pre_hook_handle = talker.register_forward_pre_hook(on_forward_pre)
        hook_handle = talker.register_forward_hook(on_forward)
        try:
            model.model.generate(**prepared.generate_kwargs)
        except GenerationCancelled:
            events.put(("stopped", None))
        except Exception as exc:
            events.put(("error", str(exc)))
        finally:
            pre_hook_handle.remove()
            hook_handle.remove()
            events.put(("done", None))

    def _decode_codes_to_audio_blocking(
        self,
        model,
        generated_codes: torch.Tensor,
        ref_code: Optional[torch.Tensor],
        prepend_ref_code: bool,
    ) -> tuple[np.ndarray, int]:
        if generated_codes.numel() == 0:
            return np.zeros(0, dtype=np.float32), self.sampling_rate_for_mode("voice_clone")

        decode_codes = generated_codes
        ref_len = 0
        if prepend_ref_code and ref_code is not None and ref_code.numel() > 0:
            ref_len = int(ref_code.shape[0])
            decode_codes = torch.cat([ref_code, generated_codes], dim=0)

        wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": decode_codes}])
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)

        if ref_len > 0:
            total_len = int(decode_codes.shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]

        return wav, int(sample_rate)

    async def stream_pcm16(self, params: StreamParams, stop_event: threading.Event):
        model = await self.ensure_model(params.mode)
        expected_sr = self.sampling_rate_for_mode(params.mode)
        decode_hop = self.decode_hop_for_mode(params.mode)
        loop = asyncio.get_running_loop()
        prepared = await loop.run_in_executor(
            None,
            self._build_stream_request_blocking,
            model,
            params,
        )
        self._attach_stop_criteria(prepared, stop_event)

        stream_lock = self._stream_locks[params.mode]
        async with stream_lock:
            events: Queue = Queue()
            generation_future = loop.run_in_executor(
                None,
                self._run_generate_with_hook_blocking,
                model,
                prepared,
                events,
                stop_event,
            )

            decoded_global_frames = 0
            frame_base_global = 0
            frame_buffer: list[torch.Tensor] = []
            done = False

            try:
                while True:
                    if stop_event.is_set() and done:
                        break

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
                        elif event == "eos":
                            done = True
                        elif event == "stopped":
                            done = True
                        elif event == "done":
                            done = True

                    end_global = frame_base_global + len(frame_buffer)
                    frames_ready = end_global - decoded_global_frames
                    should_decode = (
                        frames_ready >= STREAM_DECODE_EVERY_FRAMES
                        or (done and frames_ready > 0)
                    )

                    if should_decode:
                        start_global = max(
                            frame_base_global,
                            end_global - STREAM_MAX_DECODE_FRAMES,
                            decoded_global_frames - STREAM_LOOKBACK_FRAMES,
                        )
                        start_local = start_global - frame_base_global
                        if start_local < 0:
                            start_local = 0
                        codes = torch.stack(frame_buffer[start_local:], dim=0)
                        prepend_ref_code = start_global == 0
                        wav_f32, sample_rate = await loop.run_in_executor(
                            None,
                            self._decode_codes_to_audio_blocking,
                            model,
                            codes,
                            prepared.ref_code,
                            prepend_ref_code,
                        )
                        local_total_frames = max(1, end_global - start_global)
                        emit_from_frames = max(0, decoded_global_frames - start_global)
                        emit_from_samples = int(round((emit_from_frames / local_total_frames) * wav_f32.size))
                        if emit_from_frames > 0 and emit_from_samples == 0:
                            emit_from_samples = min(wav_f32.size, emit_from_frames * decode_hop)
                        if emit_from_samples < 0:
                            emit_from_samples = 0
                        if emit_from_samples > wav_f32.size:
                            emit_from_samples = wav_f32.size
                        wav_delta = wav_f32[emit_from_samples:]
                        decoded_global_frames = end_global

                        if sample_rate != expected_sr:
                            logger.warning(
                                "Decoded sample rate (%s) differs from expected sample rate (%s).",
                                sample_rate,
                                expected_sr,
                            )

                        if wav_delta.size > 0:
                            delta = wav_delta
                            pcm16 = np.clip(delta * 32767.0, -32768, 32767).astype(np.int16)
                            if pcm16.size > 0:
                                yield pcm16.tobytes()

                        keep_from_global = max(
                            frame_base_global,
                            decoded_global_frames - STREAM_LOOKBACK_FRAMES,
                        )
                        drop = keep_from_global - frame_base_global
                        if drop > 0:
                            frame_buffer = frame_buffer[drop:]
                            frame_base_global = keep_from_global

                    if done and decoded_global_frames >= end_global:
                        break
                    if not had_event:
                        await asyncio.sleep(STREAM_POLL_SECONDS)
            finally:
                stop_event.set()
                try:
                    await asyncio.wait_for(generation_future, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Generation thread did not stop within timeout.")


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
