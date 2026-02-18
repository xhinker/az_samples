#!/usr/bin/env python3
"""Standalone minimal Qwen3-TTS streaming demo (no audio_gen.py dependency)."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


# Edit for your blog demo.
MODEL_ID = "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice_main"
TEXT = "Hello! This is Qwen3-TTS true streaming. Audio is produced chunk by chunk from codec frames."
LANGUAGE = "Auto"
SPEAKER = "vivian"
INSTRUCTION = ""
OUT_WAV = Path("outputs/qwen3_tts_stream_demo.wav")


# Core streaming decode knobs (same idea as audio_gen.py).
DECODE_EVERY_FRAMES = 12
DECODE_OVERLAP_FRAMES = 24
MAX_DECODE_FRAMES = 144
POLL_SECONDS = 0.01
DEFAULT_DECODE_HOP = 1920


def _normalize_speaker(model: Qwen3TTSModel, speaker: str) -> str:
    getter = getattr(model, "get_supported_speakers", None)
    if not callable(getter):
        return speaker
    supported = getter() or []
    if not supported:
        return speaker
    normalized = (speaker or "").strip().lower()
    if not normalized:
        return str(supported[0]).lower()
    supported_set = {str(s).lower() for s in supported}
    if normalized not in supported_set:
        raise ValueError(f"Unsupported speaker: {speaker!r}, choose from {sorted(supported_set)}")
    return normalized


def _build_generate_kwargs(
    model: Qwen3TTSModel,
    text: str,
    language: str,
    speaker: str,
    instruction: str,
) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("Text is required.")

    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    languages = [language or "Auto"]
    model._validate_languages(languages)

    speakers = [_normalize_speaker(model, speaker)]
    model._validate_speakers(speakers)

    instruct_ids = [None]
    if instruction.strip():
        instruct_ids = [model._tokenize_texts([model._build_instruct_text(instruction.strip())])[0]]

    gen_kwargs = model._merge_generate_kwargs(non_streaming_mode=False)
    return {
        "input_ids": input_ids,
        "instruct_ids": instruct_ids,
        "languages": languages,
        "speakers": speakers,
        "non_streaming_mode": False,
        **gen_kwargs,
    }


def _decode_codes(model: Qwen3TTSModel, codes: torch.Tensor) -> tuple[np.ndarray, int]:
    if codes.ndim == 0:
        codes = codes.view(1, 1)
    elif codes.ndim == 1:
        codes = codes.view(-1, 1)
    elif codes.ndim > 2:
        codes = codes.reshape(codes.shape[0], -1)
    wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": codes}])
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    return wav, int(sample_rate)


def stream_custom_voice_pcm16(
    model: Qwen3TTSModel,
    text: str,
    language: str = LANGUAGE,
    speaker: str = SPEAKER,
    instruction: str = INSTRUCTION,
):
    """Yield PCM16 chunks by hooking into talker forward pass (true streaming)."""
    generate_kwargs = _build_generate_kwargs(model, text, language, speaker, instruction)
    decode_hop = DEFAULT_DECODE_HOP
    hop_getter = getattr(model.model.speech_tokenizer, "get_decode_upsample_rate", None)
    if callable(hop_getter):
        decode_hop = max(1, int(hop_getter()))

    eos_token_id = int(model.model.config.talker_config.codec_eos_token_id)
    talker = model.model.talker
    events: Queue = Queue()

    def on_forward(_module, _inputs, out):
        hidden_states = getattr(out, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            return
        codec_ids = hidden_states[-1]
        if codec_ids is None:
            return
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

    def run_generate():
        try:
            model.model.generate(**generate_kwargs)
        except Exception as exc:
            events.put(("error", str(exc)))
        finally:
            events.put(("done", None))

    hook_handle = talker.register_forward_hook(on_forward)
    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()

    frame_buffer = torch.empty((0, 0), dtype=torch.long)
    decoded_frames = 0
    done = False

    try:
        while True:
            had_event = False
            while True:
                try:
                    event, payload = events.get_nowait()
                except Empty:
                    break
                had_event = True

                if event == "frame":
                    payload_2d = payload
                    if frame_buffer.numel() == 0:
                        frame_buffer = payload_2d.clone()
                    else:
                        if payload_2d.shape[1] != frame_buffer.shape[1]:
                            expected_width = int(frame_buffer.shape[1])
                            if expected_width > 0 and payload_2d.numel() % expected_width == 0:
                                payload_2d = payload_2d.reshape(-1, expected_width)
                            else:
                                continue
                        frame_buffer = torch.cat([frame_buffer, payload_2d], dim=0)
                elif event == "error":
                    raise RuntimeError(payload)
                elif event in {"eos", "done"}:
                    done = True

            end_frames = int(frame_buffer.shape[0])
            frames_ready = end_frames - decoded_frames
            should_decode = frames_ready >= DECODE_EVERY_FRAMES or (done and frames_ready > 0)

            if should_decode:
                decode_end = min(end_frames, decoded_frames + MAX_DECODE_FRAMES)
                start = max(0, decoded_frames - DECODE_OVERLAP_FRAMES)
                if decode_end > start:
                    codes = frame_buffer[start:decode_end]
                    wav_f32, _sr = _decode_codes(model, codes)
                    local_total = max(1, decode_end - start)
                    emitted_local_frames = max(0, decoded_frames - start)
                    emit_from = int(emitted_local_frames * decode_hop)
                    if emit_from > wav_f32.size:
                        emit_from = int(round((emitted_local_frames / local_total) * wav_f32.size))
                    emit_from = max(0, min(emit_from, wav_f32.size))
                    wav_delta = wav_f32[emit_from:]
                    decoded_frames = decode_end
                    if wav_delta.size > 0:
                        pcm16 = np.clip(wav_delta * 32767.0, -32768, 32767).astype(np.int16)
                        if pcm16.size > 0:
                            yield pcm16.tobytes()

            if done and decoded_frames >= end_frames:
                break
            if not had_event:
                time.sleep(POLL_SECONDS)
    finally:
        hook_handle.remove()
        thread.join(timeout=5.0)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    sample_rate = int(getattr(model, "sampling_rate", 24000))

    start = time.perf_counter()
    first_chunk_latency = None
    all_chunks: list[np.ndarray] = []

    for idx, chunk in enumerate(stream_custom_voice_pcm16(model, TEXT), start=1):
        if first_chunk_latency is None:
            first_chunk_latency = time.perf_counter() - start
            print(f"first chunk latency: {first_chunk_latency:.2f}s")
        arr = np.frombuffer(chunk, dtype=np.int16).copy()
        all_chunks.append(arr)
        print(f"chunk {idx}: {arr.size} samples ({arr.size / sample_rate * 1000:.1f} ms)")

    if not all_chunks:
        raise RuntimeError("No audio chunks generated.")

    audio_i16 = np.concatenate(all_chunks)
    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT_WAV), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

    elapsed = time.perf_counter() - start
    print(f"saved: {OUT_WAV}")
    print(f"stream time: {elapsed:.2f}s")
    print(f"audio duration: {audio_i16.size / sample_rate:.2f}s")


if __name__ == "__main__":
    main()
