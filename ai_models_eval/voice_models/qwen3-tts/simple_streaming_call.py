#!/usr/bin/env python3
"""Minimal Qwen3-TTS streaming demo (custom voice mode)."""

from __future__ import annotations

import asyncio
import threading
import time
import wave
from pathlib import Path

import numpy as np

from audio_gen import (
    DEFAULT_CUSTOM_MODEL_ID,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER,
    QwenStreamingService,
    StreamParams,
    torch,
)


# Edit these 3 values for your blog demo.
TEXT = "Hello! This is a minimal Qwen3-TTS streaming demo. Audio comes chunk by chunk."
MODEL_ID = DEFAULT_CUSTOM_MODEL_ID
OUT_WAV = Path("outputs/qwen3_tts_stream_demo.wav")


async def main() -> None:
    device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    service = QwenStreamingService(
        custom_model_id=MODEL_ID,
        voice_clone_model_id=MODEL_ID,
        device=device,
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )

    # Load once so we can normalize speaker name for this model.
    model = await service.ensure_model("custom_voice")
    speaker = service.normalize_speaker(DEFAULT_SPEAKER, model)

    params = StreamParams(
        text=TEXT,
        mode="custom_voice",
        language=DEFAULT_LANGUAGE,
        speaker=speaker,
        instruction="",
        reference_audio_bytes=None,
        reference_text="",
    )

    stop_event = threading.Event()
    sample_rate = service.sampling_rate_for_mode("custom_voice")
    all_chunks: list[np.ndarray] = []
    start = time.perf_counter()
    first_chunk_s: float | None = None

    try:
        idx = 0
        async for pcm_bytes in service.stream_pcm16(params, stop_event):
            idx += 1
            if first_chunk_s is None:
                first_chunk_s = time.perf_counter() - start
                print(f"first chunk latency: {first_chunk_s:.2f}s")

            chunk_i16 = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
            all_chunks.append(chunk_i16)
            chunk_ms = (chunk_i16.size / sample_rate) * 1000.0
            print(f"chunk {idx}: {chunk_i16.size} samples ({chunk_ms:.1f} ms)")
    finally:
        stop_event.set()

    if not all_chunks:
        raise RuntimeError("No audio chunks were generated.")

    audio_i16 = np.concatenate(all_chunks)
    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT_WAV), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

    total_s = time.perf_counter() - start
    audio_s = audio_i16.size / sample_rate
    print(f"saved: {OUT_WAV}")
    print(f"total stream time: {total_s:.2f}s")
    print(f"audio duration: {audio_s:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
