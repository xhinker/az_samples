#!/usr/bin/env python3
"""Audio generation and streaming for Higgs Audio v3 TTS (transformers port).

Architecture:
  - Uses multimodalart/higgs-audio-v3-tts-4b-transformers port (trust_remote_code)
  - HiggsMultimodalQwen3ForConditionalGeneration (Qwen3-4B backbone + multi-codebook head)
  - 8 codebooks, delay pattern, 24 kHz output
  - Streaming: runs the AR loop, periodically decodes partial code rows to audio,
    yields PCM16 chunks in real time.
"""

from __future__ import annotations

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

# Monkey-patch to skip the caching allocator warmup that can cause OOM
# when GPU memory is limited. Safe to remove when GPU has enough free memory.
_original_warmup = modeling_utils.caching_allocator_warmup
modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

logger = logging.getLogger("higgs_tts_server")

DEFAULT_MODEL_PATH = "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"

# Streaming decode constants
STREAM_DECODE_EVERY_FRAMES = 12
STREAM_DECODE_OVERLAP_FRAMES = 24
STREAM_MAX_DECODE_FRAMES = 144

_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# Codec-vocab specials (from the Higgs model internals)
BOC_ID = 1024
EOC_ID = 1025


# --------------------------------------------------------------------------- #
# Delay pattern helpers (inlined from modeling_higgs_multimodal_qwen3 to avoid
# relative-import issues when loading via trust_remote_code).
# These are pure-Torch functions — no external dependencies.
# --------------------------------------------------------------------------- #

def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """``[T, N]`` raw codes -> ``[T + N - 1, N]`` delayed, BOC/EOC padded."""
    T, N = codes_TN.shape
    out = torch.full(
        (T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype
    )
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """``[L, N]`` delayed (L >= N) -> ``[L - (N - 1), N]`` raw codes."""
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
    """Per-request delay/EOC sampler state machine."""
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
    """One AR step of the multi-codebook delay sampler. Mutates ``state``."""
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
# End inlined helpers
# --------------------------------------------------------------------------- #


def _detect_lang(text: str) -> str:
    stripped = text.replace(" ", "")
    if not stripped:
        return "en"
    cjk = len(_CJK_CHAR_RE.findall(stripped))
    return "zh" if cjk / len(stripped) > 0.15 else "en"


def estimate_max_new_tokens(text: str) -> int:
    """Estimate max_new_tokens needed for a text."""
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


class HiggsStreamingService:
    """Manages Higgs Audio v3 TTS model lifecycle and streaming generation."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        max_gpu_memory: Optional[str] = None,
    ):
        """Initialize the service.

        Args:
            model_path: Path to higgs-audio-v3-tts-4b model directory.
            device: Target GPU (e.g. "cuda:1"). Used as preferred device for
                balanced placement across all available GPUs + CPU offload.
            dtype: "bfloat16" or "float16".
            max_gpu_memory: If set (e.g. "4.5GiB"), use that limit per GPU with
                CPU offload. If None (default), use device_map="balanced" to
                auto-distribute across all GPUs, falling back to the target
                GPU-only placement if balanced fails.
        """
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.max_gpu_memory = max_gpu_memory
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self._stream_lock = asyncio.Lock()
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
            for stop_event in self._active_stop_events.values():
                stop_event.set()
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
            logger.info(
                "Loaded Higgs Audio v3 TTS model from %s (dtype=%s)",
                self.model_path, self.dtype_str,
            )
            return model, tokenizer

    def _build_max_memory(self) -> dict:
        """Build a max_memory dict for balanced device placement.

        Prioritizes the target GPU, then distributes remaining capacity to
        other GPUs and CPU offload.
        """
        n_gpus = torch.cuda.device_count()
        if not n_gpus:
            raise RuntimeError("No CUDA devices available")

        target_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
        # Clamp to actual GPU count
        target_id = min(target_id, n_gpus - 1)

        mem_map = {}
        for i in range(n_gpus):
            total_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            if i == target_id:
                # Reserve most of the target GPU but leave headroom
                mem_map[i] = f"{min(total_gb - 2, total_gb * 0.85):.0f}GiB"
            elif i != target_id:
                # Use available GPUs for overflow
                mem_map[i] = f"{max(total_gb - 4, total_gb * 0.3):.0f}GiB"

        mem_map["cpu"] = "60GiB"
        return mem_map

    def _load_model(self):
        import os
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if self.max_gpu_memory:
            # Explicit memory limit — use auto device_map with CPU offload
            device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            max_memory = {device_id: self.max_gpu_memory, "cpu": "60GiB"}
            logger.info("Loading model with max_gpu_memory=%s on GPU %d",
                        self.max_gpu_memory, device_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=self.dtype,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="/tmp/higgs_offload",
            ).eval()
        else:
            # Balanced placement across all GPUs (handles multi-GPU setups where
            # some GPUs are partially occupied by other processes).
            max_memory = self._build_max_memory()
            logger.info("Loading model with balanced placement: max_memory=%s",
                        {k: v for k, v in max_memory.items() if isinstance(k, int)})

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    dtype=self.dtype,
                    device_map="balanced",
                    max_memory=max_memory,
                    offload_folder="/tmp/higgs_offload",
                ).eval()
                logger.info("Balanced placement succeeded")
            except Exception as balanced_err:
                logger.warning("Balanced placement failed (%s), falling back to direct placement",
                              balanced_err)
                # Fallback: try direct placement on target GPU
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    dtype=self.dtype,
                    device_map=self.device,
                ).eval()

        # Pre-load the codec
        try:
            codec = model.get_audio_codec()
            logger.info("Audio codec pre-loaded: %s", type(codec).__name__)
        except Exception as e:
            logger.warning("Audio codec warmup had issue (will retry on first use): %s", e)

        return model, tokenizer

    async def stream_pcm16(
        self,
        params: StreamParams,
        stop_event: threading.Event,
    ) -> AsyncIterator[bytes]:
        """Generate speech and stream PCM16LE bytes in real time.

        Uses the Higgs AR loop: prefill → autoregressive code generation →
        periodic decode → yield PCM16 chunks.
        """
        model, tokenizer = await self.ensure_model()
        loop = asyncio.get_running_loop()
        pcm_queue: asyncio.Queue = asyncio.Queue()

        def _run_generation():
            try:
                N = model.num_codebooks
                text = params.text.strip()

                # ── Reference audio encoding ──
                delayed_ref = None
                if params.reference_audio_bytes is not None:
                    with io.BytesIO(params.reference_audio_bytes) as buf:
                        with wave.open(buf, 'rb') as wf:
                            sr = wf.getframerate()
                            n_frames = wf.getnframes()
                            audio_data = np.frombuffer(
                                wf.readframes(n_frames), dtype=np.int16
                            )
                            audio_float = audio_data.astype(np.float32) / 32768.0
                    ref_tensor = torch.from_numpy(audio_float)
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

                # ── Streaming AR loop ──
                decoded_pcm_samples = 0

                for step in range(max_tokens):
                    if stop_event.is_set():
                        break

                    logits_NV = model.audio_head(hidden_last).to(torch.float32)[0]
                    codes_N = _sampler_step(
                        logits_NV, state,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        top_k=params.top_k,
                    )
                    rows.append(codes_N.cpu())
                    if state.generation_done:
                        break

                    # Periodically decode partial results
                    frames_ready = len(rows)
                    if frames_ready >= N and (frames_ready % STREAM_DECODE_EVERY_FRAMES == 0 or state.generation_done):
                        try:
                            delayed_LN = torch.stack(rows, dim=0)
                            codes_TN = reverse_delay_pattern(delayed_LN)
                            if codes_TN.shape[0] > 0:
                                wav = model._decode_codes(codes_TN)
                                wav_np = wav.numpy()
                                new_samples = wav_np[decoded_pcm_samples:]
                                if new_samples.size > 0:
                                    pcm16 = np.clip(
                                        new_samples * 32767.0, -32768, 32767
                                    ).astype(np.int16)
                                    if pcm16.size > 0:
                                        asyncio.run_coroutine_threadsafe(
                                            pcm_queue.put(pcm16.tobytes()), loop
                                        )
                                decoded_pcm_samples = len(wav_np)
                        except ValueError:
                            pass

                    # Forward step for next token
                    step_embed = model.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
                    cache_pos = torch.tensor([position], device=model.device)
                    out = model.model(
                        inputs_embeds=step_embed.to(inputs_embeds.dtype),
                        past_key_values=past,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    past = out.past_key_values
                    hidden_last = out.last_hidden_state[:, -1, :]
                    position += 1

                # ── Final decode of remaining frames ──
                if rows and not stop_event.is_set():
                    try:
                        delayed_LN = torch.stack(rows, dim=0)
                        codes_TN = reverse_delay_pattern(delayed_LN)
                        if codes_TN.shape[0] > 0:
                            wav = model._decode_codes(codes_TN)
                            wav_np = wav.numpy()
                            new_samples = wav_np[decoded_pcm_samples:]
                            if new_samples.size > 0:
                                pcm16 = np.clip(
                                    new_samples * 32767.0, -32768, 32767
                                ).astype(np.int16)
                                if pcm16.size > 0:
                                    asyncio.run_coroutine_threadsafe(
                                        pcm_queue.put(pcm16.tobytes()), loop
                                    )
                    except ValueError:
                        pass

            except Exception as exc:
                logger.exception("Generation failed")
                asyncio.run_coroutine_threadsafe(
                    pcm_queue.put(("error", str(exc))), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(
                    pcm_queue.put(("done", None)), loop
                )

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
