#!/usr/bin/env python3
"""Audio generation and streaming for Higgs Audio v3 TTS (transformers port).

Architecture:
  - Uses bosonai/higgs-audio-v3-tts-4b with trust_remote_code
  - HiggsMultimodalQwen3ForConditionalGeneration (Qwen3-4B backbone + multi-codebook head)
  - 8 codebooks, delay pattern, 24 kHz output
  - Streaming: decode-all-from-scratch each cycle (consistent boundaries), emit only
    genuinely new samples. Overlap crossfade blending at chunk boundaries to smooth
    neural codec artifacts.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio
import io
import logging
import re
import threading
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np
import torch
import transformers.modeling_utils as modeling_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

_original_warmup = modeling_utils.caching_allocator_warmup
modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

logger = logging.getLogger("higgs_tts_server")

DEFAULT_MODEL_PATH = "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"

# Streaming decode constants
STREAM_DECODE_EVERY_FRAMES = 40   # Higher = smoother (fewer decode boundaries), but more latency
CROSSFADE_SAMPLES = 640    # ~27ms crossfade for smoother boundaries between larger chunks    # ~13ms overlap crossfade to smooth chunk boundaries
CROSSFADE_FRAMES = 8       # Extra frames of context for crossfade


_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

BOC_ID = 1024
EOC_ID = 1025


# --------------------------------------------------------------------------- #
# Delay pattern helpers (inlined)
# --------------------------------------------------------------------------- #

def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    T, N = codes_TN.shape
    out = torch.full((T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype)
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        raise ValueError(f"delayed rows L={L} < num_codebooks N={N}")
    out = torch.empty((T, N), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


@dataclass
class _SamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: torch.Tensor | None = None


def _sample(logits_NV: torch.Tensor, temperature: float, top_p: float | None,
            top_k: int | None) -> torch.Tensor:
    if temperature <= 1e-5:
        return logits_NV.argmax(dim=-1)
    logits = logits_NV / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_idx, remove)
        logits = torch.where(scatter, float("-inf"), logits)
    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1)


def _sampler_step(logits_NV: torch.Tensor, state: _SamplerState, *,
                  temperature: float, top_p: float | None,
                  top_k: int | None) -> torch.Tensor:
    N = state.num_codebooks
    codes_N = _sample(logits_NV, temperature, top_p, top_k).to(torch.long)

    if state.delay_count < N:
        next_cb = state.delay_count + 1
        if next_cb < N:
            codes_N[next_cb:] = BOC_ID
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_N[0].item()) == EOC_ID:
        if N <= 2:
            state.generation_done = True
        else:
            state.eoc_countdown = N - 2

    if not state.generation_done:
        state.last_codes = codes_N.clone()
    return codes_N


# --------------------------------------------------------------------------- #

def _detect_lang(text: str) -> str:
    stripped = text.replace(" ", "")
    if not stripped:
        return "en"
    cjk = len(_CJK_CHAR_RE.findall(stripped))
    return "zh" if cjk / len(stripped) > 0.15 else "en"


def estimate_max_new_tokens(text: str) -> int:
    words = len(text.split())
    chars = len(text)
    estimated = max(int(words * 4.0), int(chars * 0.8)) + 160
    return max(192, min(4096, estimated))


@dataclass
class StreamParams:
    text: str
    reference_audio_bytes: Optional[bytes] = None
    reference_sample_rate: int = 24000
    reference_text: Optional[str] = None
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None


def _crossfade(prev_tail: np.ndarray, curr_head: np.ndarray) -> np.ndarray:
    """Crossfade blend two overlapping audio segments to eliminate clicks."""
    fade_len = min(len(prev_tail), len(curr_head))
    if fade_len <= 1:
        return curr_head
    # Smooth cosine crossfade: prev*fade_out + curr*fade_in
    t = np.linspace(0, 1, fade_len)
    fade_out = 0.5 * (1 - np.cos(np.pi * t))   # ramps 0→1 → multiply with (1-fade) = 1→0
    fade_in = 0.5 * (1 + np.cos(np.pi * t))     # ramps 1→0 → but we want 0→1
    # Actually use simple linear for speed, cosine for quality:
    fade_out = np.sin(np.linspace(0, np.pi/2, fade_len)) ** 2   # 0→1
    fade_in = np.cos(np.linspace(0, np.pi/2, fade_len)) ** 2    # 1→0
    blended = prev_tail * (1 - fade_out) + curr_head * fade_out
    return np.concatenate([blended, curr_head[fade_len:]])


class HiggsStreamingService:
    """Manages Higgs Audio v3 TTS model lifecycle and streaming generation."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda:2", dtype: str = "bfloat16"):
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self._active_stop_events: dict[int, threading.Event] = {}
        self._active_stop_events_lock = threading.Lock()
        self._stop_counter = 0

    @property
    def sample_rate(self) -> int:
        return 24000

    def register_active_stream(self, stop_event: threading.Event) -> int:
        with self._active_stop_events_lock:
            self._stop_counter += 1
            sid = self._stop_counter
            self._active_stop_events[sid] = stop_event
            return sid

    def clear_active_stream(self, sid: int) -> None:
        with self._active_stop_events_lock:
            self._active_stop_events.pop(sid, None)

    def request_stop(self) -> int:
        with self._active_stop_events_lock:
            count = len(self._active_stop_events)
            for ev in self._active_stop_events.values():
                ev.set()
            if count > 0:
                logger.info("Stop requested for %d active streams", count)
            return count

    async def ensure_model(self):
        if self._model is not None:
            return self._model, self._tokenizer
        async with self._lock:
            if self._model is not None:
                return self._model, self._tokenizer
            loop = asyncio.get_running_loop()
            model, tokenizer = await loop.run_in_executor(None, self._load_model)
            self._model = model
            self._tokenizer = tokenizer
            logger.info("Loaded Higgs Audio v3 TTS on %s (dtype=%s)", self.device, self.dtype_str)
            try:
                dev_idx = int(self.device.split(":")[-1])
                free_gb = torch.cuda.mem_get_info(dev_idx)[0] / 1e9
                total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / 1e9
                logger.info("GPU %s memory: %.1f/%.0f GB used", self.device, total_gb - free_gb, total_gb)
            except Exception:
                pass
            return model, tokenizer

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        logger.info("Loading model directly on %s ...", self.device)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, dtype=self.dtype,
            device_map=self.device,
        ).eval()
        try:
            codec = model.get_audio_codec()
            logger.info("Audio codec loaded on %s: %s", codec.device, type(codec).__name__)
        except Exception as e:
            logger.warning("Audio codec warmup issue (will retry on first use): %s", e)
        return model, tokenizer

    async def stream_pcm16(self, params: StreamParams, stop_event: threading.Event) -> AsyncIterator[bytes]:
        """Generate speech and stream PCM16LE bytes in real time.

        Decode-all-from-scratch each cycle for consistent neural vocoder output.
        Track sample-level position precisely. Crossfade-blend chunk boundaries
        to smooth any remaining codec artifacts.
        """
        model, tokenizer = await self.ensure_model()
        loop = asyncio.get_running_loop()
        pcm_queue: asyncio.Queue = asyncio.Queue()

        def _run_generation():
            try:
                N = model.num_codebooks
                device = model.device
                text = params.text.strip()

                # ── Reference audio encoding ──
                delayed_ref = None
                if params.reference_audio_bytes is not None:
                    with io.BytesIO(params.reference_audio_bytes) as buf:
                        with wave.open(buf, 'rb') as wf:
                            sr = wf.getframerate()
                            n_frames = wf.getnframes()
                            audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                            audio_float = audio_data.astype(np.float32) / 32768.0
                    ref_tensor = torch.from_numpy(audio_float).to(device)
                    codes_TN = model._encode_reference(ref_tensor, sr)
                    delayed_ref = apply_delay_pattern(codes_TN.cpu())

                # ── Prompt assembly ──
                prompt_ids = model._build_prompt_ids(
                    tokenizer, text,
                    num_ref_tokens=0 if delayed_ref is None else delayed_ref.shape[0],
                    reference_text=params.reference_text,
                )

                # ── Prefill ──
                inputs_embeds = model._prefill_embeds(prompt_ids, delayed_ref)
                out = model.model(inputs_embeds=inputs_embeds, use_cache=True)
                past = out.past_key_values
                hidden_last = out.last_hidden_state[:, -1, :]
                position = inputs_embeds.shape[1]

                state = _SamplerState(num_codebooks=N)
                rows: list[torch.Tensor] = []
                max_tokens = min(estimate_max_new_tokens(text), 2048)

                # ── Streaming state ──
                decoded_pcm_samples = 0
                prev_chunk_tail: Optional[np.ndarray] = None   # Tail of previous chunk for crossfade
                first_chunk = True

                def _emit_chunk():
                    nonlocal decoded_pcm_samples, prev_chunk_tail, first_chunk
                    delayed_LN = torch.stack(rows, dim=0)
                    codes_TN = reverse_delay_pattern(delayed_LN)
                    if codes_TN.shape[0] < N:
                        return
                    wav = model._decode_codes(codes_TN)
                    wav_np = wav.numpy().astype(np.float32)
                    
                    # Only emit samples we haven't sent yet
                    new_samples = wav_np[decoded_pcm_samples:]
                    if len(new_samples) == 0:
                        return
                    
                    decoded_pcm_samples = len(wav_np)
                    
                    # Crossfade with previous chunk tail to smooth boundaries
                    if not first_chunk and prev_chunk_tail is not None and len(prev_chunk_tail) >= CROSSFADE_SAMPLES:
                        overlap = new_samples[:CROSSFADE_SAMPLES]
                        blended = _crossfade(prev_chunk_tail[-CROSSFADE_SAMPLES:], overlap)
                        new_samples = np.concatenate([blended, new_samples[CROSSFADE_SAMPLES:]])
                    
                    first_chunk = False
                    
                    if len(new_samples) > 0:
                        pcm16 = np.clip(new_samples * 32767.0, -32768, 32767).astype(np.int16)
                        asyncio.run_coroutine_threadsafe(pcm_queue.put(pcm16.tobytes()), loop)

                    # Keep tail of this chunk for next crossfade
                    if len(new_samples) > CROSSFADE_SAMPLES:
                        prev_chunk_tail = new_samples[-CROSSFADE_SAMPLES:].copy()

                # ── Streaming AR loop ──
                for step in range(max_tokens):
                    if stop_event.is_set():
                        break

                    logits_NV = model.audio_head(hidden_last).to(torch.float32)[0]
                    codes_N = _sampler_step(
                        logits_NV, state,
                        temperature=params.temperature, top_p=params.top_p, top_k=params.top_k,
                    )
                    rows.append(codes_N.cpu())
                    if state.generation_done:
                        break

                    frames_ready = len(rows)
                    if frames_ready >= N and (frames_ready % STREAM_DECODE_EVERY_FRAMES == 0 or state.generation_done):
                        try:
                            _emit_chunk()
                        except Exception as e:
                            logger.warning("Decode error (continuing): %s", e)

                    # Forward step
                    step_embed = model.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
                    cache_pos = torch.tensor([position], device=device)
                    out = model.model(
                        inputs_embeds=step_embed.to(inputs_embeds.dtype),
                        past_key_values=past, use_cache=True, cache_position=cache_pos,
                    )
                    past = out.past_key_values
                    hidden_last = out.last_hidden_state[:, -1, :]
                    position += 1

                # ── Final decode: emit remaining frames ──
                if rows and not stop_event.is_set():
                    try:
                        _emit_chunk()
                    except Exception as e:
                        logger.warning("Final decode error (continuing): %s", e)

            except Exception as exc:
                logger.exception("Generation failed")
                asyncio.run_coroutine_threadsafe(pcm_queue.put(("error", str(exc))), loop)
            finally:
                asyncio.run_coroutine_threadsafe(pcm_queue.put(("done", None)), loop)

        future = loop.run_in_executor(None, _run_generation)
        try:
            while True:
                item = await pcm_queue.get()
                if isinstance(item, tuple):
                    tag, payload = item
                    if tag == "done":
                        break
                    elif tag == "error":
                        raise RuntimeError(payload)
                else:
                    yield item
        finally:
            if not future.done():
                stop_event.set()
            try:
                await future
            except Exception:
                pass
