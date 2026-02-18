#!/usr/bin/env python3
"""Audio generation and streaming internals for Qwen3-TTS server."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import tempfile
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Optional

import numpy as np
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


logger = logging.getLogger("qwen3_tts_server")

DEFAULT_CUSTOM_MODEL_ID = (
    "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice_main"
)
DEFAULT_VOICE_CLONE_MODEL_ID = (
    "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main"
)
DEFAULT_SPEAKER = "vivian"
DEFAULT_LANGUAGE = "Auto"
STREAM_DECODE_EVERY_FRAMES = 12
STREAM_DECODE_OVERLAP_FRAMES = 24
STREAM_MAX_DECODE_FRAMES = 144
STREAM_POLL_SECONDS = 0.01
STREAM_BUFFER_RESET_SECONDS = 30.0
SEGMENT_CROSSFADE_SECONDS = 0.05
DEFAULT_DECODE_HOP = 1920
TEXT_REANCHOR_MAX_WORDS = 90
TEXT_REANCHOR_TARGET_SECONDS = 30.0

_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SPLIT_HINT_CHARS = " ,;，；。！？!?、"


class StopEventCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


class GenerationCancelled(Exception):
    pass


def split_text_for_reanchor(text: str, max_words: int = TEXT_REANCHOR_MAX_WORDS) -> list[str]:
    clean = " ".join(text.strip().split())
    if not clean:
        return []

    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", clean) if s]
    segments: list[str] = []

    def estimate_seconds(piece: str) -> float:
        normalized = " ".join(piece.strip().split())
        if not normalized:
            return 0.0
        non_space_chars = len(normalized.replace(" ", ""))
        words = len(normalized.split())
        cjk_chars = len(_CJK_CHAR_RE.findall(normalized))
        if cjk_chars >= max(8, int(non_space_chars * 0.20)):
            # CJK speech is better approximated by character count.
            return (cjk_chars / 5.5) + ((non_space_chars - cjk_chars) / 14.0)
        return max(words / 2.8, non_space_chars / 15.0)

    def split_oversized_piece(piece: str) -> list[str]:
        normalized = " ".join(piece.strip().split())
        if not normalized:
            return []
        if estimate_seconds(normalized) <= TEXT_REANCHOR_TARGET_SECONDS:
            return [normalized]

        non_space_chars = len(normalized.replace(" ", ""))
        cjk_chars = len(_CJK_CHAR_RE.findall(normalized))
        is_cjk_heavy = cjk_chars >= max(8, int(non_space_chars * 0.20))
        chars_per_second = 6 if is_cjk_heavy else 15
        window = max(80, int(TEXT_REANCHOR_TARGET_SECONDS * chars_per_second))

        parts: list[str] = []
        start = 0
        text_len = len(normalized)
        while start < text_len:
            end = min(text_len, start + window)
            if end < text_len:
                cut = -1
                scan_start = max(start + int(window * 0.5), start + 1)
                for idx in range(end, scan_start - 1, -1):
                    if normalized[idx - 1] in _SPLIT_HINT_CHARS:
                        cut = idx
                        break
                if cut == -1:
                    cut = end
            else:
                cut = end
            part = normalized[start:cut].strip()
            if part:
                parts.append(part)
            start = cut
        return parts

    current = ""

    def flush_current():
        nonlocal current
        if current:
            segments.append(current)
            current = ""

    for sentence in sentences:
        for piece in split_oversized_piece(sentence):
            words = len(piece.split())
            if words > max_words:
                flush_current()
                start = 0
                word_list = piece.split()
                while start < len(word_list):
                    part_words = word_list[start:start + max_words]
                    segments.append(" ".join(part_words))
                    start += max_words
                continue

            candidate = piece if not current else f"{current} {piece}"
            if current and estimate_seconds(candidate) > TEXT_REANCHOR_TARGET_SECONDS:
                flush_current()
                current = piece
            else:
                current = candidate

    flush_current()
    return segments if segments else [clean]


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
        device = self.device
        # Qwen3-TTS uses a custom talker generate that does not support tensors
        # being split across multiple CUDA devices.  When device_map="auto" is
        # requested and more than one GPU is visible, pin the whole model to
        # cuda:0 so all tensors stay on a single device.
        if (
            device == "auto"
            and torch is not None
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            device = "cuda:0"
            logger.info(
                "Multiple CUDA devices detected (%d GPUs). Pinning TTS model to cuda:0 "
                "to avoid cross-device tensor errors. Use --device to override.",
                torch.cuda.device_count(),
            )

        kwargs = {
            "device_map": device,
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
                        frame = codec_ids[0].detach().to("cpu", dtype=torch.long)
                        if frame.ndim == 0:
                            frame = frame.view(1, 1)
                        elif frame.ndim == 1:
                            frame = frame.view(1, -1)
                        elif frame.ndim > 2:
                            frame = frame.reshape(frame.shape[0], -1)
                        if frame.numel() == 0:
                            return
                        if int(frame[0, 0].item()) == eos_token_id:
                            events.put(("eos", None))
                            return
                        events.put(("frame", frame.contiguous()))
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
        ref_trim_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        if generated_codes.numel() == 0:
            return np.zeros(0, dtype=np.float32), self.sampling_rate_for_mode("voice_clone")

        decode_codes = generated_codes
        if decode_codes.ndim == 0:
            decode_codes = decode_codes.view(1, 1)
        elif decode_codes.ndim == 1:
            decode_codes = decode_codes.view(-1, 1)
        elif decode_codes.ndim > 2:
            decode_codes = decode_codes.reshape(decode_codes.shape[0], -1)

        ref_len = 0
        if prepend_ref_code and ref_code is not None and ref_code.numel() > 0:
            ref_codes = ref_code.detach()
            if ref_codes.ndim == 0:
                ref_codes = ref_codes.view(1, 1)
            elif ref_codes.ndim == 1:
                if decode_codes.ndim == 2 and decode_codes.shape[1] > 0 and ref_codes.numel() % decode_codes.shape[1] == 0:
                    ref_codes = ref_codes.view(-1, decode_codes.shape[1])
                else:
                    ref_codes = ref_codes.view(-1, 1)
            elif ref_codes.ndim > 2:
                ref_codes = ref_codes.reshape(ref_codes.shape[0], -1)

            if ref_codes.ndim == 2 and decode_codes.ndim == 2 and ref_codes.shape[1] != decode_codes.shape[1]:
                if ref_codes.shape[0] == decode_codes.shape[1]:
                    ref_codes_t = ref_codes.transpose(0, 1).contiguous()
                    if ref_codes_t.shape[1] == decode_codes.shape[1]:
                        ref_codes = ref_codes_t
                if ref_codes.shape[1] != decode_codes.shape[1]:
                    raise RuntimeError(
                        f"Reference/code dimensions mismatch: ref_code={tuple(ref_codes.shape)}, generated={tuple(decode_codes.shape)}"
                    )

            ref_len = int(ref_codes.shape[0])
            decode_codes = torch.cat([ref_codes, decode_codes], dim=0)

        wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": decode_codes}])
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)

        if ref_len > 0:
            if ref_trim_samples is not None and ref_trim_samples >= 0:
                cut = int(ref_trim_samples)
            else:
                total_len = int(decode_codes.shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            if cut < 0:
                cut = 0
            if cut > wav.shape[0]:
                cut = wav.shape[0]
            wav = wav[cut:]

        return wav, int(sample_rate)

    def _decode_ref_prefix_samples_blocking(
        self,
        model,
        ref_code: Optional[torch.Tensor],
    ) -> tuple[int, int]:
        if ref_code is None or ref_code.numel() == 0:
            return 0, self.sampling_rate_for_mode("voice_clone")
        wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": ref_code}])
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return int(wav.size), int(sample_rate)

    async def _stream_prepared_pcm16(
        self,
        model,
        prepared: PreparedStreamRequest,
        expected_sr: int,
        decode_hop: int,
        stop_event: threading.Event,
    ):
        loop = asyncio.get_running_loop()
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
        frame_buffer = torch.empty((0, 0), dtype=torch.long)
        ref_trim_samples = 0
        done = False
        reset_anchor_global = 0
        emitted_samples_since_reset = 0
        reset_samples_threshold = max(1, int(round(expected_sr * STREAM_BUFFER_RESET_SECONDS)))
        decode_hop_samples = max(1, int(decode_hop))
        has_ref_code = prepared.ref_code is not None and prepared.ref_code.numel() > 0
        needs_ref_prepend = has_ref_code
        if prepared.ref_code is not None and prepared.ref_code.numel() > 0:
            ref_trim_samples, ref_sr = await loop.run_in_executor(
                None,
                self._decode_ref_prefix_samples_blocking,
                model,
                prepared.ref_code,
            )
            if ref_sr != expected_sr:
                logger.warning(
                    "Reference decode sample rate (%s) differs from expected sample rate (%s).",
                    ref_sr,
                    expected_sr,
                )

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
                        if payload.numel() > 0:
                            payload_2d = payload
                            if payload_2d.ndim == 0:
                                payload_2d = payload_2d.view(1, 1)
                            elif payload_2d.ndim == 1:
                                payload_2d = payload_2d.view(1, -1)
                            elif payload_2d.ndim > 2:
                                payload_2d = payload_2d.reshape(payload_2d.shape[0], -1)

                            if frame_buffer.numel() == 0:
                                frame_buffer = payload_2d.contiguous().clone()
                            else:
                                if payload_2d.shape[1] != frame_buffer.shape[1]:
                                    expected_width = int(frame_buffer.shape[1])
                                    if expected_width > 0 and payload_2d.numel() % expected_width == 0:
                                        payload_2d = payload_2d.reshape(-1, expected_width)
                                    else:
                                        logger.warning(
                                            "Dropping misaligned codec payload shape=%s expected_width=%s",
                                            tuple(payload_2d.shape),
                                            expected_width,
                                        )
                                        continue
                                frame_buffer = torch.cat([frame_buffer, payload_2d], dim=0)
                    elif event == "error":
                        raise RuntimeError(payload)
                    elif event == "eos":
                        done = True
                    elif event == "stopped":
                        done = True
                    elif event == "done":
                        done = True

                end_global = frame_base_global + int(frame_buffer.shape[0])
                frames_ready = end_global - decoded_global_frames
                should_decode = (
                    frames_ready >= STREAM_DECODE_EVERY_FRAMES
                    or (done and frames_ready > 0)
                )
                if frames_ready > STREAM_DECODE_EVERY_FRAMES * 6:
                    logger.warning(
                        "Decode backlog is high (%s frames); streaming quality may degrade if GPU is saturated.",
                        frames_ready,
                    )

                if should_decode:
                    decode_end_global = min(
                        end_global,
                        decoded_global_frames + STREAM_MAX_DECODE_FRAMES,
                    )
                    start_global = max(
                        frame_base_global,
                        decoded_global_frames - STREAM_DECODE_OVERLAP_FRAMES,
                    )
                    start_local = start_global - frame_base_global
                    if start_local < 0:
                        start_local = 0

                    end_local = decode_end_global - frame_base_global
                    if end_local <= start_local:
                        if not had_event:
                            await asyncio.sleep(STREAM_POLL_SECONDS)
                        continue

                    codes = frame_buffer[start_local:end_local]
                    prepend_ref_code = (
                        has_ref_code
                        and (needs_ref_prepend or start_global == reset_anchor_global)
                    )
                    wav_f32, sample_rate = await loop.run_in_executor(
                        None,
                        self._decode_codes_to_audio_blocking,
                        model,
                        codes,
                        prepared.ref_code,
                        prepend_ref_code,
                        ref_trim_samples if prepend_ref_code else None,
                    )
                    local_total_frames = max(1, decode_end_global - start_global)
                    emitted_local_frames = max(0, decoded_global_frames - start_global)
                    emit_from_samples = int(emitted_local_frames * decode_hop_samples)
                    if emit_from_samples > wav_f32.size:
                        # Keep continuity if decode window/sample mapping drifts slightly.
                        emit_from_samples = int(
                            round((emitted_local_frames / local_total_frames) * wav_f32.size)
                        )
                    if emit_from_samples < 0:
                        emit_from_samples = 0
                    if emit_from_samples > wav_f32.size:
                        emit_from_samples = wav_f32.size
                    wav_delta = wav_f32[emit_from_samples:]
                    decoded_global_frames = decode_end_global

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
                            emitted_samples_since_reset += int(wav_delta.size)
                            yield pcm16.tobytes()

                    keep_from_global = max(
                        frame_base_global,
                        decoded_global_frames - STREAM_DECODE_OVERLAP_FRAMES,
                    )
                    drop = keep_from_global - frame_base_global
                    if drop > 0:
                        frame_buffer = frame_buffer[drop:].contiguous()
                        frame_base_global = keep_from_global

                    if prepend_ref_code:
                        needs_ref_prepend = False

                    if (
                        emitted_samples_since_reset >= reset_samples_threshold
                        and decoded_global_frames >= end_global
                    ):
                        # Periodically reset bookkeeping while preserving overlap context,
                        # so decoder continuity does not get hard-cut every reset cycle.
                        overlap_keep = max(0, min(
                            STREAM_DECODE_OVERLAP_FRAMES,
                            decoded_global_frames - frame_base_global,
                        ))
                        if overlap_keep > 0:
                            frame_buffer = frame_buffer[-overlap_keep:].contiguous()
                            frame_base_global = decoded_global_frames - overlap_keep
                        else:
                            empty_width = int(frame_buffer.shape[1]) if frame_buffer.ndim == 2 else 0
                            frame_buffer = torch.empty((0, empty_width), dtype=torch.long)
                            frame_base_global = decoded_global_frames

                        reset_anchor_global = decoded_global_frames
                        emitted_samples_since_reset = 0
                        needs_ref_prepend = has_ref_code
                        logger.info(
                            "Reset streaming decode state after %.1fs; preserved %s overlap frames.",
                            STREAM_BUFFER_RESET_SECONDS,
                            overlap_keep,
                        )

                if done and decoded_global_frames >= end_global:
                    break
                if not had_event:
                    await asyncio.sleep(STREAM_POLL_SECONDS)
        finally:
            try:
                await asyncio.wait_for(generation_future, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Generation thread did not stop within timeout.")

    async def stream_pcm16(self, params: StreamParams, stop_event: threading.Event):
        model = await self.ensure_model(params.mode)
        expected_sr = self.sampling_rate_for_mode(params.mode)
        loop = asyncio.get_running_loop()
        segments = split_text_for_reanchor(params.text, max_words=TEXT_REANCHOR_MAX_WORDS)
        if not segments:
            raise ValueError("Text is required.")
        if len(segments) > 1:
            logger.info("Long text detected. Re-anchoring stream into %s segments.", len(segments))

        stream_lock = self._stream_locks[params.mode]
        async with stream_lock:
            crossfade_samples = max(0, int(round(expected_sr * SEGMENT_CROSSFADE_SECONDS)))
            pending_tail = np.zeros(0, dtype=np.int16)
            for idx, segment_text in enumerate(segments, start=1):
                if stop_event.is_set():
                    break
                segment_params = StreamParams(
                    text=segment_text,
                    mode=params.mode,
                    language=params.language,
                    speaker=params.speaker,
                    instruction=params.instruction,
                    reference_audio_bytes=params.reference_audio_bytes,
                    reference_text=params.reference_text,
                )
                prepared = await loop.run_in_executor(
                    None,
                    self._build_stream_request_blocking,
                    model,
                    segment_params,
                )
                self._attach_stop_criteria(prepared, stop_event)
                if len(segments) > 1:
                    logger.info("Streaming segment %s/%s", idx, len(segments))
                is_last_segment = idx == len(segments)
                keep_samples = crossfade_samples if (crossfade_samples > 0 and not is_last_segment) else 0

                segment_iter = self._stream_prepared_pcm16(
                    model=model,
                    prepared=prepared,
                    expected_sr=expected_sr,
                    decode_hop=self.decode_hop_for_mode(params.mode),
                    stop_event=stop_event,
                )

                head_remainder = np.zeros(0, dtype=np.int16)
                prefetched_arrays: list[np.ndarray] = []

                if pending_tail.size > 0 and crossfade_samples > 0:
                    head = np.zeros(0, dtype=np.int16)
                    while head.size < crossfade_samples:
                        try:
                            chunk = await anext(segment_iter)
                        except StopAsyncIteration:
                            break
                        arr = np.frombuffer(chunk, dtype=np.int16).copy()
                        if arr.size == 0:
                            continue
                        need = crossfade_samples - head.size
                        take = min(need, arr.size)
                        if take > 0:
                            head = np.concatenate([head, arr[:take]])
                        rest = arr[take:]
                        if rest.size > 0:
                            prefetched_arrays.append(rest)

                    if head.size == 0:
                        if is_last_segment:
                            yield pending_tail.tobytes()
                            pending_tail = np.zeros(0, dtype=np.int16)
                    else:
                        blend_len = min(pending_tail.size, head.size)
                        if pending_tail.size > blend_len:
                            yield pending_tail[:-blend_len].tobytes()
                        if blend_len > 0:
                            fade_in = np.linspace(0.0, 1.0, num=blend_len, endpoint=True, dtype=np.float32)
                            fade_out = 1.0 - fade_in
                            blended = (
                                pending_tail[-blend_len:].astype(np.float32) * fade_out
                                + head[:blend_len].astype(np.float32) * fade_in
                            )
                            blended_i16 = np.clip(np.round(blended), -32768, 32767).astype(np.int16)
                            if blended_i16.size > 0:
                                yield blended_i16.tobytes()
                        head_remainder = head[blend_len:]
                        pending_tail = np.zeros(0, dtype=np.int16)

                holdback = np.zeros(0, dtype=np.int16)

                async def emit_arr(arr: np.ndarray):
                    nonlocal holdback
                    if arr.size == 0:
                        return
                    combined = arr if holdback.size == 0 else np.concatenate([holdback, arr])
                    if keep_samples <= 0:
                        holdback = np.zeros(0, dtype=np.int16)
                        yield combined.tobytes()
                        return
                    if combined.size <= keep_samples:
                        holdback = combined
                        return
                    emit_part = combined[:-keep_samples]
                    holdback = combined[-keep_samples:]
                    if emit_part.size > 0:
                        yield emit_part.tobytes()

                async for out in emit_arr(head_remainder):
                    yield out
                for prefetched in prefetched_arrays:
                    async for out in emit_arr(prefetched):
                        yield out

                async for pcm_chunk in segment_iter:
                    arr = np.frombuffer(pcm_chunk, dtype=np.int16).copy()
                    async for out in emit_arr(arr):
                        yield out

                if keep_samples > 0:
                    pending_tail = holdback
                elif holdback.size > 0:
                    yield holdback.tobytes()

            if pending_tail.size > 0 and not stop_event.is_set():
                yield pending_tail.tobytes()
