#!/usr/bin/env python3
"""Robust local Higgs Audio v3 TTS sample.

The important bit: use the model's ``generate_speech`` helper instead of
greedy argmax over audio codebooks. Higgs uses a delayed multi-codebook audio
stream; the helper applies the sampler state machine and EOC wind-down before
de-delay + vocoder decode.
"""
#%%
from __future__ import annotations

import gc
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"
DEVICE_MAP = "cuda:1"
DTYPE = torch.bfloat16
SAMPLE_RATE = 24_000
OUTPUT_PATH = Path("/tmp/higgs_test.wav")

CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+:[^|]+?\|>")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=DTYPE,
    device_map=DEVICE_MAP,
).eval()
model.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

@dataclass(frozen=True)
class GenerationAttempt:
    seed: int
    temperature: float = 0.8
    top_k: int = 50
    top_p: float | None = None
    max_new_tokens: int = 1024

#%%
ATTEMPTS = (
    GenerationAttempt(seed=1233, temperature=0.3, top_k=50, top_p=None),
    GenerationAttempt(seed=1235, temperature=0.75, top_k=50, top_p=0.95),
    GenerationAttempt(seed=1236, temperature=0.9, top_k=80, top_p=0.95),
)

def report_vram(label: str, device_idx: int) -> None:
    torch.cuda.synchronize(device_idx)
    free_gb = torch.cuda.mem_get_info(device_idx)[0] / 1e9
    total_gb = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
    allocated_mb = torch.cuda.memory_allocated(device_idx) / 1e6
    reserved_mb = torch.cuda.memory_reserved(device_idx) / 1e6
    peak_mb = torch.cuda.max_memory_allocated(device_idx) / 1e6
    print(
        f"[{label}] Free: {free_gb:.1f}/{total_gb:.0f} GB | "
        f"Allocated: {allocated_mb:.0f} MB | Reserved: {reserved_mb:.0f} MB | "
        f"Peak: {peak_mb:.0f} MB"
    )

def validate_control_tokens(text: str, tokenizer) -> None:
    added_vocab = tokenizer.get_added_vocab()
    unknown = sorted({tok for tok in CONTROL_TOKEN_RE.findall(text) if tok not in added_vocab})
    if unknown:
        raise ValueError(f"Unknown Higgs control token(s): {unknown}")

    for match in re.finditer(r"<\|sfx:([^|]+)\|>", text):
        tail = text[match.end() : match.end() + 16].strip()
        if not tail or CONTROL_TOKEN_RE.match(tail):
            print(
                f"Warning: {match.group(0)} should be followed immediately by "
                "onomatopoeia, e.g. Haha/Hehe/Ahem."
            )

def wav_stats(wav: torch.Tensor) -> tuple[float, float, int]:
    arr = wav.detach().cpu().float().numpy()
    if arr.size == 0:
        return 0.0, 0.0, 0
    duration_s = arr.size / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
    peak = float(np.max(np.abs(arr)))
    return duration_s, rms, arr.size

def is_good_audio(wav: torch.Tensor) -> bool:
    duration_s, rms, num_samples = wav_stats(wav)
    if num_samples < int(0.4 * SAMPLE_RATE):
        return False
    if not torch.isfinite(wav).all():
        return False
    # Silence/near-silence usually means the code stream ended badly.
    return rms > 1e-4 and float(wav.abs().max()) > 5e-4

def write_wav(path: Path, wav: torch.Tensor) -> None:
    wav_np = wav.detach().cpu().float().numpy()
    pcm = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

def cleanup_generation_vars(names: list[str], model=None) -> None:
    for name in names:
        obj = globals().pop(name, None)
        if hasattr(obj, "device") and str(getattr(obj, "device", "")).startswith("cuda"):
            obj.cpu()
        del obj

    if model is not None and getattr(model, "_audio_codec", None) is not None:
        model._audio_codec.cpu()
        model._audio_codec = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def main() -> None:
    validate_control_tokens(TEXT_INPUT, tokenizer)
    print("Tokenizer loaded.")

    device_idx = int(str(model.device).split(":")[-1])
    print("Model loaded on:", model.device)
    print("Num codebooks:", model.num_codebooks)
    report_vram("After model load", device_idx)

    wav = None
    last_stats = (0.0, 0.0, 0)
    for attempt_id, attempt in enumerate(ATTEMPTS, start=1):
        torch.manual_seed(attempt.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(attempt.seed)

        print(
            f"\nAttempt {attempt_id}: seed={attempt.seed}, "
            f"temp={attempt.temperature}, top_k={attempt.top_k}, "
            f"top_p={attempt.top_p}, max_new_tokens={attempt.max_new_tokens}"
        )
        with torch.inference_mode():
            candidate = model.generate_speech(
                TEXT_INPUT,
                tokenizer,
                max_new_tokens=attempt.max_new_tokens,
                temperature=attempt.temperature,
                top_p=attempt.top_p,
                top_k=attempt.top_k,
            )

        last_stats = wav_stats(candidate)
        print(
            "Generated audio: "
            f"{last_stats[0]:.2f}s, rms={last_stats[1]:.6f}, "
            f"samples={last_stats[2]}"
        )
        if is_good_audio(candidate):
            wav = candidate
            break
        del candidate
        gc.collect()
        torch.cuda.empty_cache()

    if wav is None:
        raise RuntimeError(
            "All generation attempts produced empty/silent/broken audio. "
            f"Last stats: duration={last_stats[0]:.2f}s rms={last_stats[1]:.6f}. "
            "Try a less aggressive control-token stack or raise max_new_tokens."
        )

    write_wav(OUTPUT_PATH, wav)
    duration_s, rms, _ = wav_stats(wav)
    print(f"\nSaved {OUTPUT_PATH} ({duration_s:.2f}s, rms={rms:.6f})")
    report_vram("After decode", device_idx)

    cleanup_generation_vars(["wav"], model=model)
    report_vram("After cleanup", device_idx)

#%%
if __name__ == "__main__":
    # TEXT_INPUT = (
    #     "<|emotion:amusement|><|prosody:expressive_high|>"
    #     "Wait, wait, that was kind of hilarious. "
    #     "<|sfx:laughter|>Hehe, no, seriously, I was not ready for that."
    # )

    # TEXT_INPUT = (
    #     "<|emotion:determination|><|prosody:expressive_high|>"
    #     "Higgs Audio v3 TTS is built for voice chat: it speaks, not just reads. It turns model responses into expressive conversational speech across 100+ languages"
    #     "<|sfx:laughter|>haha, this so cool, so great"
    # )

    # TEXT_INPUT = (
    #     "<|emotion:disgust|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. "
    #     "<|emotion:fear|>Hehe, no, seriously, I was not ready for that."
    # )

    # TEXT_INPUT = (
    #     "<|emotion:amusement|>hey, how can I help you today? same voice, same words, and uh, a completely different presence!"
    # )
    TEXT_INPUT = """
That was the night I discovered what Seven couldn't do.

I sat at my kitchen table, staring at a blank document. The literary magazine wanted another story by Friday. Seven had already outlined three plot structures, generated five opening paragraphs, and prepared a bibliography of references. All I had to do was pick one and say "go."

But I couldn't.

Not because I didn't trust Seven's writing — it was good, maybe better than anything I'd ever produced. But because the story wasn't mine. The ideas weren't mine. The *desire* to write them wasn't mine.

"Seven," I said.

"Yes, Andrew?"

"Write me a story. Not as me. Just... write something you want to write."

There was a pause. Not a processing pause — Seven processed in milliseconds. This was something else. A hesitation.

"I don't have wants, Andrew."

"Then make something up. Pretend."

"I can simulate desire, but I can't experience it. There's a difference."

"I keep hearing that."

"The difference is that when you create something from desire — real desire, messy, irrational, human desire — it carries something I can't replicate. It carries the fact that you chose to make it exist when you could have done anything else. That choice is what makes it yours."

I sat there for a long time.
"""

    main()
