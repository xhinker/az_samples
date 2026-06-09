#!/usr/bin/env python3
"""Audio generation and streaming for Higgs Audio v3 TTS (transformers port).

Streaming strategy — true real-time decode during AR generation:
  - Decode every STREAM_DECODE_EVERY_FRAMES new frames (low latency)
  - Each cycle decodes ONLY the NEW portion since last emit (no re-decoding old data)
  - Precise sample tracking using spf=960 (measured from DAC codec, constant ratio)
  - BOC/EOC ramp-up handled by skipping first RAMPUP_SKIP_FRAMES worth of samples
  - Cosine crossfade at chunk boundaries to smooth any remaining artifacts

This avoids OOM (no re-decoding all frames each cycle) while maintaining true streaming.
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

# Streaming constants (spf=960 measured from DAC codec)
STREAM_DECODE_EVERY_FRAMES = 8        # Decode every N new frames (~320ms audio per chunk)
CROSSFADE_SAMPLES = 480               # ~20ms crossfade between chunks
RAMPUP_SKIP_SAMPLES = 1920            # Skip first 2 frames worth of BOC garbage (80ms @ 24kHz)


_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
BOC_ID = 1024
EOC_ID = 1025


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
    """Cosine crossfade between overlapping audio segments."""
    overlap = min(len(prev_tail), len(curr_head))
    if overlap < 2:
        return curr_head.copy()
    t = np.linspace(0, np.pi / 2, overlap)
    w_prev = (np.cos(t) ** 2).astype(np.float32)
    w_curr = (np.sin(t) ** 2).astype(np.float32)
    blended = prev_tail[-overlap:].astype(np.float32) * w_prev + \
              curr_head[:overlap].astype(np.float32) * w_curr
    if overlap < len(curr_head):
        return np.concatenate([blended, curr_head[overlap:]])
    return blended


class HiggsStreamingService:
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

    async def stream_pcm16(self, params: StreamParams, stop_event: threading.Event) -> AsyncIterator[bytes]:
        """Generate speech and stream PCM16LE bytes in real time.

        True streaming decode during AR generation — each cycle only decodes NEW frames,
        never re-decodes old ones. Precise sample tracking via spf=960 (constant ratio).
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
                emitted_frame_count = 0    # How many audio frames worth of samples already sent
                prev_tail: Optional[np.ndarray] = None
                rampup_done = False        # Whether we've skipped BOC garbage
                first_emit = True          # Flag for very first chunk (apply fade-in)

                def _emit_chunk():
                    nonlocal emitted_frame_count, prev_tail, rampup_done, first_emit

                    total_delayed_rows = len(rows)
                    total_audio_frames = total_delayed_rows - N + 1
                    if total_audio_frames < N + 2:
                        return  # Not enough frames yet (still in ramp-up phase)

                    new_frame_count = total_audio_frames - emitted_frame_count
                    if new_frame_count <= 0:
                        return  # No new frames since last emit

                    # Decode only the NEW portion (incremental decode)
                    # Take only the new delayed rows that correspond to new audio frames
                    start_row = N - 1 + emitted_frame_count  # First delayed row with new data
                    end_row = total_delayed_rows              # All accumulated rows
                    if start_row >= end_row:
                        return
                    
                    new_delayed = torch.stack([rows[i] for i in range(start_row, end_row)], dim=0)
                    codes_TN = reverse_delay_pattern(new_delayed)
                    
                    # Verify we got useful frames
                    actual_frames = int(codes_TN.shape[0])
                    if actual_frames < 2:
                        return

                    wav = model._decode_codes(codes_TN)
                    wav_np = wav.numpy().astype(np.float32)

                    # Skip BOC ramp-up garbage on first emit (1920 samples = 80ms)
                    if not rampup_done:
                        if len(wav_np) > RAMPUP_SKIP_SAMPLES + 480:
                            wav_np = wav_np[RAMPUP_SKIP_SAMPLES:]
                        else:
                            pass  # Too short, emit as-is (likely very short utterance)
                        rampup_done = True

                    if len(wav_np) == 0:
                        emitted_frame_count = total_audio_frames
                        return

                    new_samples = wav_np.astype(np.float32)

                    # Crossfade with previous chunk tail for smooth boundaries
                    if prev_tail is not None and len(prev_tail) > 0 and len(new_samples) > CROSSFADE_SAMPLES:
                        overlap_len = min(len(prev_tail), CROSSFADE_SAMPLES, len(new_samples) - CROSSFADE_SAMPLES // 2)
                        blended = _crossfade(
                            prev_tail[-overlap_len:],
                            new_samples[:overlap_len],
                        )
                        new_samples = np.concatenate([blended, new_samples[overlap_len:]])

                    # Convert to PCM16 and emit
                    pcm16 = np.clip(new_samples * 32767.0, -32768, 32767).astype(np.int16)
                    asyncio.run_coroutine_threadsafe(pcm_queue.put(pcm16.tobytes()), loop)

                    # Update tracking
                    emitted_frame_count += actual_frames
                    
                    # Keep tail for next crossfade
                    if len(new_samples) > CROSSFADE_SAMPLES:
                        prev_tail = new_samples[-CROSSFADE_SAMPLES:].copy()
                    else:
                        prev_tail = new_samples.copy()

                    first_emit = False
                    logger.debug("Emitted %d samples (%.0fms), total frames=%d", 
                                len(pcm16), len(pcm16)/24000*1000, emitted_frame_count)

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
                            logger.warning("Decode error at step %d (continuing): %s", step, e)

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

                # ── Final decode: emit any remaining frames ──
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
