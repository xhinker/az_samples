"""
audio_gen.py - Real streaming TTS generation engine for IndexTTS2

Real streaming: generated tokens and decoded voice chunks are emitted segment
by segment during the model's forward function — not after full completion.

Usage:
    from audio_gen import get_engine, SAMPLE_RATE

    engine = get_engine()
    await engine.load_model()

    # Stream raw PCM int16 bytes chunk by chunk
    async for pcm_bytes in engine.stream_tts(text, "/path/to/reference.wav"):
        # each chunk is raw PCM int16, 22050 Hz, mono
        process(pcm_bytes)

    # Or get the full WAV file at once
    wav_bytes = await engine.generate_wav(text, "/path/to/reference.wav")
"""

import os
import sys
import io
import wave
import struct
import asyncio
import logging
import threading
import concurrent.futures as cf
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the IndexTTS repo to Python path
REPO_DIR = os.path.join(BASE_DIR, "repos", "index-tts")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Model weights (as specified in requirements)
MODEL_DIR = os.path.join(
    BASE_DIR, "models", "tts_hf_models", "IndexTeam", "IndexTTS-2_main"
)
MODEL_CFG_PATH = os.path.join(MODEL_DIR, "config.yaml")

# HuggingFace model cache
HF_CACHE_DIR = os.path.join(BASE_DIR, "checkpoints", "hf_cache")

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes per sample


# ---------------------------------------------------------------------------
# PCM / WAV helpers
# ---------------------------------------------------------------------------

def tensor_to_pcm_bytes(wav_tensor: torch.Tensor) -> bytes:
    """
    Convert a wav tensor to raw int16 PCM bytes.

    The tensor from infer_generator is float32, values scaled to [-32767, 32767]
    via torch.clamp(32767 * wav, ...).  Shape is typically [1, T] (mono).
    """
    if wav_tensor is None or wav_tensor.numel() == 0:
        return b""
    wav = wav_tensor.detach().cpu()
    if wav.ndim > 1:
        wav = wav.flatten()
    return wav.to(torch.int16).numpy().tobytes()


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM int16 data in a complete WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def build_streaming_wav_header(sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Build a 44-byte WAV header with open-ended (0xFFFFFFFF) size fields.

    This is the standard trick for streaming WAV: the total data length is
    unknown, so we set both RIFF size and data chunk size to 0xFFFFFFFF.
    Clients that support streaming WAV (including Web Audio API) accept this.
    """
    bits_per_sample = 16
    byte_rate = sample_rate * CHANNELS * bits_per_sample // 8
    block_align = CHANNELS * bits_per_sample // 8
    UNKNOWN = 0xFFFFFFFF

    header  = struct.pack("<4sI4s", b"RIFF", UNKNOWN, b"WAVE")
    header += struct.pack(
        "<4sIHHIIHH",
        b"fmt ", 16,        # fmt chunk size (PCM)
        1,                  # audio format: PCM
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    header += struct.pack("<4sI", b"data", UNKNOWN)
    return header  # exactly 44 bytes


# ---------------------------------------------------------------------------
# TTS Engine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Manages the IndexTTS2 model and exposes an async streaming TTS interface.

    A single-worker ThreadPoolExecutor keeps all blocking PyTorch calls off
    the asyncio event loop.  Audio chunks are relayed from the inference
    thread to async consumers via an asyncio.Queue.
    """

    def __init__(self, use_fp16: bool = True, use_cuda_kernel: bool = False, device: str = "cuda:0"):
        self._model = None
        self._use_fp16 = use_fp16
        self._use_cuda_kernel = use_cuda_kernel
        self._device = device
        # One worker: GPU can only run one inference at a time
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts_worker"
        )
        self._model_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_sync(self):
        """Load the model on the inference thread (blocking)."""
        if self._model is not None:
            return self._model

        # Must set HF cache and cwd before importing infer_v2, because
        # infer_v2.py sets HF_HUB_CACHE at module level using a relative path.
        os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
        os.chdir(BASE_DIR)

        logger.info("Loading IndexTTS2 model from %s", MODEL_DIR)
        from indextts.infer_v2 import IndexTTS2

        self._model = IndexTTS2(
            cfg_path=MODEL_CFG_PATH,
            model_dir=MODEL_DIR,
            use_fp16=self._use_fp16,
            use_cuda_kernel=self._use_cuda_kernel,
            device=self._device,
        )
        logger.info("IndexTTS2 model loaded successfully")
        return self._model

    async def load_model(self):
        """Ensure the model is loaded — async-safe and idempotent."""
        async with self._model_lock:
            if self._model is None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, self._load_model_sync)

    # ------------------------------------------------------------------
    # Core streaming generator
    # ------------------------------------------------------------------

    async def stream_tts(
        self,
        text: str,
        spk_audio_prompt: str,
        quick_streaming_tokens: int = 5,
        max_text_tokens_per_segment: int = 40,
        diffusion_steps: int = 20,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """
        Real streaming TTS async generator.

        Yields raw PCM int16 bytes (mono, 22050 Hz) as each text segment is
        decoded — the first audio arrives before the full text is synthesized.

        Latency-tuning knobs:
            quick_streaming_tokens:      Segments totalling fewer than this many
                                         tokens are NOT merged together, so the
                                         very first natural sentence stays small.
                                         Lower = faster first chunk. (default 5)
            max_text_tokens_per_segment: Hard cap on segment length; shorter
                                         segments produce audio sooner.
                                         Lower = faster first chunk, more passes.
                                         (default 40 for streaming; non-streaming
                                          uses the model default of 120)
            diffusion_steps:             s2mel flow-matching steps per segment.
                                         Fewer steps = faster, slightly lower
                                         quality. 10-15 is a good streaming
                                         trade-off. (default 15)

        Args:
            text:             Text to synthesise.
            spk_audio_prompt: Absolute path to the reference voice audio.
            **kwargs:         Forwarded to infer_generator (temperature,
                              top_p, max_mel_tokens, etc.).

        Yields:
            bytes: Raw PCM int16 audio chunk.
        """
        await self.load_model()

        loop = asyncio.get_event_loop()
        # Small queue: less buffering = faster stop response
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        stop_event = threading.Event()
        SENTINEL = object()

        def _run_inference():
            """
            Inference thread.  Uses stop_event to exit cleanly when the
            consumer (client) disconnects or clicks Stop.

            queue.put() is polled with a 200 ms timeout so the thread
            can notice stop_event even when the queue is full and the
            async consumer has already stopped draining it.
            """
            gen = None
            try:
                gen = self._model.infer_generator(
                    spk_audio_prompt=spk_audio_prompt,
                    text=text,
                    output_path=None,
                    stream_return=True,
                    quick_streaming_tokens=quick_streaming_tokens,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    diffusion_steps=diffusion_steps,
                    **kwargs,
                )
                for wav_chunk in gen:
                    # Check stop signal at the top of every segment loop
                    if stop_event.is_set():
                        break
                    if not (isinstance(wav_chunk, torch.Tensor) and wav_chunk.numel() > 0):
                        continue
                    pcm = tensor_to_pcm_bytes(wav_chunk)
                    if not pcm:
                        continue
                    # Put into queue; poll every 200 ms so we catch stop_event
                    # even when the queue is full and the consumer has stopped.
                    put_fut = asyncio.run_coroutine_threadsafe(queue.put(pcm), loop)
                    while True:
                        try:
                            put_fut.result(timeout=0.2)
                            break                        # put succeeded
                        except cf.TimeoutError:
                            if stop_event.is_set():
                                put_fut.cancel()
                                return  # exit _run_inference entirely
            except Exception as exc:
                logger.exception("Error in TTS inference thread")
                if not stop_event.is_set():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(exc), loop
                        ).result(timeout=2)
                    except Exception:
                        pass
            finally:
                # Close the generator to free GPU resources immediately
                if gen is not None:
                    try:
                        gen.close()
                    except Exception:
                        pass
                # Always send SENTINEL so the consumer can exit its loop.
                # Use a timeout so we never block forever if consumer is gone.
                try:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(SENTINEL), loop
                    ).result(timeout=5)
                except Exception:
                    pass

        inference_future = loop.run_in_executor(self._executor, _run_inference)

        try:
            while True:
                chunk = await queue.get()
                if chunk is SENTINEL:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        finally:
            # 1. Tell the inference thread to stop.
            stop_event.set()
            # 2. Drain the queue so the thread is never blocked on queue.put()
            #    and can reach its own stop_event check or send SENTINEL.
            #    Loop until the thread finishes, yielding to the event loop
            #    each iteration so pending queue.put() coroutines can run.
            while not inference_future.done():
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                await asyncio.sleep(0.05)
            # Final drain (picks up the SENTINEL the thread sends in its finally)
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.debug("TTS inference thread stopped cleanly")

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    async def stream_wav(
        self,
        text: str,
        spk_audio_prompt: str,
        **kwargs,
    ) -> AsyncGenerator[bytes, None]:
        """
        Yield a streaming WAV: 44-byte header first, then raw PCM chunks.

        Compatible with HTTP chunked transfer encoding and Web Audio API.
        """
        yield build_streaming_wav_header()
        async for pcm_chunk in self.stream_tts(text, spk_audio_prompt, **kwargs):
            yield pcm_chunk

    async def generate_wav(
        self,
        text: str,
        spk_audio_prompt: str,
        **kwargs,
    ) -> bytes:
        """
        Generate complete WAV audio (waits for full synthesis).

        Returns a valid WAV file as bytes with a correct header.
        """
        pcm_chunks = []
        async for chunk in self.stream_tts(text, spk_audio_prompt, **kwargs):
            pcm_chunks.append(chunk)
        return pcm_to_wav_bytes(b"".join(pcm_chunks))


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_engine: Optional[TTSEngine] = None


def get_engine(use_fp16: bool = True, use_cuda_kernel: bool = False, device: str = "cuda:0") -> TTSEngine:
    """Return the global TTSEngine singleton, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = TTSEngine(use_fp16=use_fp16, use_cuda_kernel=use_cuda_kernel, device=device)
    return _engine
