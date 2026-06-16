#!/usr/bin/env python3
"""Higgs Audio v3 TTS — Fixed Inference with Codex patterns."""

#%% [markdown]
# # Higgs Audio v3 TTS — Improved Control Tag Handling

#%%
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import re
import struct
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import wave
import gc
import sys
import time
from pathlib import Path

# Add voice_models parent dir so we can import tts_utils
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_utils import split_text_for_reanchor, concatenate_wavs

MODEL_PATH = "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"

# ## 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Tokenizer loaded.")

# ## 2. Load Model (bfloat16, direct GPU)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda:1",
).eval()
model.requires_grad_(False)

print("Model loaded on:", model.device)
print("Num codebooks:", model.num_codebooks)

# ## 3. Helper Functions (run this cell first!)
BOC_ID = 1024
EOC_ID = 1025
LEADING_CONTROL_TOKEN_RE = re.compile(r"\s*(<\|[^|]+:[^|]+?\|>)")
TERMINAL_PUNCT_CHARS = ".!?。！？"
WEAK_PUNCT_CHARS = ",;，；、:："
CLOSING_QUOTE_CHARS = "\"'”’）)]》」』"
DEFAULT_PREROLL_TOKEN = "<|prosody:pause|>"
POST_DECODE_SILENCE_SEC = 0.03

#%%
def validate_control_tokens(text, tokenizer):
    """Validate control tokens exist in vocab."""
    # Control token validation (Codex pattern)  
    CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+:[^|]+?\|>")

    added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    unknown = sorted({tok for tok in CONTROL_TOKEN_RE.findall(text) if tok not in added_vocab})
    if unknown:
        print("WARNING: Unknown control token(s) - may not be followed well:")
        for tok in unknown[:5]:
            print("  %s" % tok)

    # Warn about SFX(Sound Effects) tags needing onomatopoeia  
    for match in re.finditer(r"<\|sfx:([^|]+)\|>", text):
        tail = text[match.end():match.end()+16].strip()
        if not tail or CONTROL_TOKEN_RE.match(tail):
            print("WARNING: %s should be followed by onomatopoeia (e.g. Haha/Ahem)" % match.group(0))

def wav_stats(wav_tensor):
    """Audio quality metrics. Returns (duration_s, rms, num_samples, peak)."""
    arr = wav_tensor.detach().cpu().float().numpy() if hasattr(wav_tensor, "detach") else wav_tensor
    if arr.size == 0:
        return 0.0, 0.0, 0, 0.0
    duration_s = arr.size / 24000.0
    rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
    peak = float(np.max(np.abs(arr)))
    return duration_s, rms, arr.size, peak

def is_good_audio(wav_tensor):
    """Check if generated audio is valid (not silent/broken/clipped)."""
    duration_s, rms, num_samples, peak = wav_stats(wav_tensor)
    if num_samples < int(0.4 * 24000):  # less than 0.4s = likely broken
        return False
    if peak < 1e-3:  # too quiet, essentially silent
        return False
    if peak >= 1.0 and rms > 0.5:  # true clipping (sustained full-scale, not just a peak)
        return False
    return rms > 1e-4

def reverse_delay_pattern(delayed_LN):
    """Undo the delay pattern applied during training."""
    L, Nc = delayed_LN.shape
    T = L - (Nc - 1)
    out = torch.empty((T, Nc), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(Nc):
        out[:, c] = delayed_LN[c : c + T, c]
    return out

def apply_delay_pattern(codes_TN):
    """Apply delay pattern to reference codes: [T, N] -> [T + N - 1, N], BOC/EOC padded."""
    T, N = codes_TN.shape
    out = torch.full((T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype)
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out

def encode_reference_audio(reference_audio, reference_sr=None):
    """Encode a reference audio file once for reuse across multiple infer() calls.

    The neural codec encoding is expensive. Encode once, reuse the result.

    Args:
        reference_audio: Path to WAV file, numpy array, or torch tensor.
        reference_sr: Sample rate (auto-detected from WAV path, default 24000).

    Returns:
        delayed_ref tensor ready to pass as reference_audio to infer().
    """
    if isinstance(reference_audio, str):
        with wave.open(reference_audio, "rb") as wf:
            ref_sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
            ref_wave = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        ref_tensor = torch.from_numpy(ref_wave).float()
    elif isinstance(reference_audio, np.ndarray):
        ref_tensor = torch.from_numpy(reference_audio).float()
        ref_sr = reference_sr or 24000
    else:
        ref_tensor = reference_audio.float()
        ref_sr = reference_sr or 24000

    with torch.inference_mode():
        codes_TN = model._encode_reference(ref_tensor, ref_sr)
    delayed_ref = apply_delay_pattern(codes_TN.cpu())
    print(f"  Reference encoded: {codes_TN.shape[0]} frames, {codes_TN.shape[1]} codebooks (delayed: {delayed_ref.shape[0]} rows)")
    clean_vram()
    return delayed_ref

def prepend_silence(wav_np, sample_rate=24000, silence_sec=POST_DECODE_SILENCE_SEC):
    """Prepend a short playback guard.

    This is only for players that clip the first samples. It cannot recover a
    missing phoneme because it runs after the neural codec has already decoded.
    """
    silence_samples = int(silence_sec * sample_rate)
    silence = np.zeros(silence_samples, dtype=wav_np.dtype)
    return np.concatenate([silence, wav_np])

def trim_trailing_silence(wav_np, sample_rate=24000, threshold=0.01, min_silence_sec=0.5):
    """Trim trailing silence/garbage from end of decoded audio.

    When model doesn't generate EOC, it keeps producing random tokens that
    decode to noise/silence. Finds last speech frame and cuts the rest.
    Only trims from the END -- never touches the start.
    """
    if len(wav_np) == 0:
        return wav_np
    abs_wav = np.abs(wav_np)
    min_samples = int(min_silence_sec * sample_rate)
    cut_point = len(wav_np)
    for i in range(len(wav_np) - 1, min_samples - 1, -1):
        if abs_wav[i] > threshold:
            cut_point = i + min_samples
            break
    if cut_point < len(wav_np):
        trimmed = len(wav_np) - cut_point
        print(f"  Trimmed {trimmed/sample_rate:.1f}s trailing silence/garbage from end")
    return wav_np[:cut_point]


def add_generation_preroll(text, preroll_token=DEFAULT_PREROLL_TOKEN):
    """Insert a model-level pause before the first spoken token.

    Higgs' codec has no left audio context at the very first generated frame.
    Starting directly on a CJK syllable can lose the attack, e.g. the beginning
    of "柳". A prosody pause in the prompt creates real generated pre-roll,
    unlike adding silence after decode.

    Leading delivery/control tokens remain first so their whole-turn behavior
    is preserved.
    """
    stripped = text.strip()
    if not stripped or stripped.startswith(preroll_token):
        return stripped

    pos = 0
    while True:
        match = LEADING_CONTROL_TOKEN_RE.match(stripped, pos)
        if not match:
            break
        pos = match.end()

    return f"{stripped[:pos]}{preroll_token}{stripped[pos:]}"

def ensure_terminal_punctuation(text):
    """Close an utterance so Higgs has a clear reason to emit EOC."""
    stripped = text.strip()
    if not stripped:
        return stripped

    suffix = ""
    body = stripped
    while body and body[-1] in CLOSING_QUOTE_CHARS:
        suffix = body[-1] + suffix
        body = body[:-1].rstrip()

    if not body or body[-1] in TERMINAL_PUNCT_CHARS:
        return body + suffix

    if body[-1] in WEAK_PUNCT_CHARS:
        body = body[:-1].rstrip()

    cjk_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", body))
    non_space_chars = len(body.replace(" ", ""))
    terminal = "。" if cjk_chars >= max(4, int(non_space_chars * 0.20)) else "."
    return body + terminal + suffix

def report_vram(device_idx=None, label=""):
    """Print VRAM diagnostics."""
    if device_idx is None:
        device_idx = int(str(model.device).split(":")[-1])

    torch.cuda.synchronize(device_idx)
    free_gb         = torch.cuda.mem_get_info(device_idx)[0] / 1e9
    total_gb        = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
    allocated_mb    = torch.cuda.memory_allocated(device_idx) / 1e6
    reserved_mb     = torch.cuda.memory_reserved(device_idx) / 1e6
    max_alloc_mb    = torch.cuda.max_memory_allocated(device_idx) / 1e6

    print(
        "[%s] Free: %.1f/%.0f GB | Alloc: %.0f MB | Reserved: %.0f MB | Peak: %.0f MB"
        % (label, free_gb, total_gb, allocated_mb, reserved_mb, max_alloc_mb)
    )

def clean_vram():
    """Aggressively reclaim VRAM after inference."""
    import gc

    if hasattr(model, "_audio_codec") and model._audio_codec is not None:
        print("  Moving cached audio codec to CPU ...")
        model._audio_codec.cpu()
        model._audio_codec = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def _sample(logits_NV, temperature, top_p, top_k):
    """Official sampling logic from modeling_higgs_multimodal_qwen3.py."""
    if temperature <= 1e-5:
        return logits_NV.argmax(dim=-1)

    logits = logits_NV / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx   = torch.sort(logits, descending=True, dim=-1)
        cum                         = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove                      = cum > top_p
        remove[..., 1:]             = remove[..., :-1].clone()
        remove[..., 0]              = False
        scatter                     = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_idx, remove)
        logits                      = torch.where(scatter, float("-inf"), logits)

    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1)

class _SamplerState:
    """Official sampler state machine (from modeling code)."""
    def __init__(self, num_codebooks):
        self.num_codebooks      = num_codebooks
        self.delay_count        = 0
        self.eoc_countdown      = None
        self.generation_done    = False

def _sampler_step(logits_NV, state, temperature, top_p, top_k):
    """One AR step of the multi-codebook delay sampler. Mutates state."""
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

    return codes_N

def infer(
    text_input,
    output_wav_path = None,
    max_steps       = None,
    temperature     = 0.8,
    top_k           = 50,
    top_p           = None,
    seed            = None,
    reference_audio = None,
    reference_sr    = None,
    reference_text  = None,
    add_preroll     = True,
    close_utterance = True,
    fail_on_cap     = False,
):
    """Run full AR generation + vocoder decode for *text_input*.

    Args:
        text_input: Text with optional control tags (emotion, prosody, style, sfx).
        output_wav_path: Path to save the output WAV (24kHz, 16-bit PCM). None = auto unique path.
        max_steps: Hard cap on AR generation steps. If None (default), auto-calculated
                   from text length: min(1024, max(192, max(words*4, chars*6) + 160)).
                   Each step is one delayed-codebook row; 25 rows ~= 1s before
                   delay reversal. EOC should stop generation before this cap.
        temperature: Sampling temperature (0.5-1.2). Lower = deterministic, higher = creative.
        top_k: Keep only top-K tokens before sampling. None = disabled.
        top_p: Nucleus sampling threshold. None = disabled.
        seed: Random seed for reproducible generation. None = random each time.
        reference_audio: Path to WAV, numpy array, or pre-encoded delayed_ref tensor (from encode_reference_audio()) for voice cloning.
                          NOTE: for speed, encode once with encode_reference_audio() and reuse the returned tensor.
                          Passing a WAV path here will re-encode the audio on every call, which is slow.
        reference_sr: Sample rate of reference audio (auto-detected from WAV if path). None = auto.
        reference_text: Transcript of reference audio (improves cloning quality). None = optional.

    Returns:
        (wav_path, duration_seconds) tuple.

    Uses official multi-codebook delay sampler with proper BOC/EOC handling.
    """

    device_idx = int(str(model.device).split(":")[-1])
    N = model.num_codebooks

    # -- dynamic max_steps: word-based (EN) and char-based (CJK), clamped [192, 4096]
    if max_steps is None:
        words = len(text_input.split())
        chars = len(text_input.replace(" ", ""))
        max_steps = min(1024, max(192, max(int(words * 4), int(chars * 6)) + 160))
        print(f"max_steps:{max_steps}")

    # -- auto unique output path if not specified
    if output_wav_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_wav_path = f"/tmp/higgs_{ts}_{id(text_input):08x}.wav"

    # -- set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -- reference audio: accept pre-encoded tensor (from encode_reference_audio) or raw input
    delayed_ref = None
    if reference_audio is not None:
        if isinstance(reference_audio, torch.Tensor):
            delayed_ref = reference_audio  # pre-encoded, reuse directly
        else:
            delayed_ref = encode_reference_audio(reference_audio, reference_sr)

    # -- build prompt embeddings ---------------------------------
    rows = []
    # report_vram(device_idx, "Before AR generation")

    source_text = ensure_terminal_punctuation(text_input) if close_utterance else text_input.strip()
    if source_text != text_input.strip():
        print("  Closed chunk with terminal punctuation for EOC stability.")

    generation_text = add_generation_preroll(source_text) if add_preroll else source_text
    if generation_text != text_input.strip():
        print("  Added model-level preroll before first spoken token.")

    num_ref_tokens = 0 if delayed_ref is None else delayed_ref.shape[0]
    prompt_ids_list = model._build_prompt_ids(
        tokenizer, generation_text, num_ref_tokens=num_ref_tokens, reference_text=reference_text
    )

    state = _SamplerState(N)

    with torch.inference_mode():
        inputs_embeds   = model._prefill_embeds(prompt_ids_list, delayed_ref)
        out             = model.model(inputs_embeds=inputs_embeds, use_cache=True)
        past            = out.past_key_values
        hidden_last     = out.last_hidden_state[:, -1, :]
        position        = inputs_embeds.shape[1]

        for step in range(max_steps):
            logits_NV = model.audio_head(hidden_last).to(torch.float32)[0]  # [N, V]
            codes_N = _sampler_step(logits_NV, state, temperature, top_p, top_k)

            if state.generation_done:
                print("EOC reached at step %d (generation complete)" % step)
                break

            rows.append(codes_N.cpu())

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

            # free per-step intermediates immediately
            del logits_NV, codes_N, step_embed, cache_pos
            position += 1

    # report_vram(device_idx, "After AR generation")
    print("Generated %d rows (codebooks), max_steps was %d" % (len(rows), max_steps))
    hit_cap = len(rows) >= max_steps - N
    if hit_cap:
        print("  WARNING: hit max_steps cap! Last sentence may be partially truncated.")
        if fail_on_cap:
            raise RuntimeError(
                f"Higgs generation hit max_steps={max_steps} before EOC for "
                f"{len(text_input)} input character(s). "
                "The chunk is probably too open-ended or too hard for this sampling seed."
            )

    if len(rows) < N:
        print("WARNING: Too few codebook steps (%d/%d). Output may be silent." % (len(rows), N))

    # -- decode codebooks -> WAV ---------------------------------
    with torch.inference_mode():
        delayed_LN = torch.stack(rows, dim=0)
        codes_TN = reverse_delay_pattern(delayed_LN)
        wav = model._decode_codes(codes_TN.to(model.device))
        wav_np = wav.numpy().astype(np.float32)
        wav_np = prepend_silence(wav_np, sample_rate=24000)
        wav_np = trim_trailing_silence(wav_np, sample_rate=24000)
        # Normalize volume to target RMS (~-16 LUFS equivalent for speech)
        current_rms = float(np.sqrt(np.mean(wav_np ** 2)))
        target_rms = 0.2  # ~-14 dBFS, typical for TTS output
        if current_rms > 0:
            gain = min(target_rms / current_rms, 5.0)  # cap gain at 5x to avoid noise amplification
            wav_np = np.clip(wav_np * gain, -1.0, 1.0)

    # write WAV file
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(24000)
        pcm = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())

    duration_s = len(wav_np) / 24000
    print("Saved %s (%.1fs audio)" % (output_wav_path, duration_s))
    
    # Audio quality metrics (Codex pattern)
    dur_s, rms, samples, peak = wav_stats(wav)
    if not is_good_audio(wav):
        print("WARNING: Poor audio quality detected! rms=%.6f, peak=%.4f" % (rms, peak))
    
    # report_vram(device_idx, "After decode")

    # cleanup intermediates (codec cleared separately below)
    for name in [
        "past", "hidden_last", "inputs_embeds", "out",
        "logits_NV", "codes_N", "step_embed", "cache_pos",
        "wav", "delayed_LN", "codes_TN", "pcm",
    ]:
        globals().pop(name, None)

    rows.clear()

    return output_wav_path, duration_s


# ======================================================================
# STREAMING INFERENCE (for AZ PAL real-time voice)
# ======================================================================

def stream_infer(
    text_input,
    max_steps       = None,
    temperature     = 0.8,
    top_k           = 50,
    top_p           = None,
    seed            = None,
    reference_audio = None,
    reference_sr    = None,
    reference_text  = None,
    add_preroll     = True,
    close_utterance = True,
    chunk_size      = 200,
):
    """Stream AR generation + vocoder decode, yielding PCM16LE bytes.

    Instead of writing to a WAV file, this generator yields raw PCM16LE
    byte chunks as soon as audio is decoded. Designed for real-time
    streaming with AZ PAL: LLM text tokens -> buffer into sentences ->
    stream_infer -> PCM bytes -> Web Audio API playback.

    Args:
        Same as infer(), plus:
        chunk_size: PCM bytes to yield per chunk (default 200 = ~4ms audio).

    Yields:
        bytes: PCM16LE audio chunks (24kHz, mono).

    Example:
        for pcm_chunk in stream_infer("Hello world.", temperature=0.8):
            audio_player.write(pcm_chunk)
    """
    device_idx = int(str(model.device).split(":")[-1])
    N = model.num_codebooks

    # -- dynamic max_steps
    if max_steps is None:
        words = len(text_input.split())
        chars = len(text_input.replace(" ", ""))
        max_steps = min(1024, max(192, max(int(words * 4), int(chars * 6)) + 160))

    # -- set random seed
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -- reference audio
    delayed_ref = None
    if reference_audio is not None:
        if isinstance(reference_audio, torch.Tensor):
            delayed_ref = reference_audio
        else:
            delayed_ref = encode_reference_audio(reference_audio, reference_sr)

    # -- build prompt embeddings
    rows = []
    source_text = ensure_terminal_punctuation(text_input) if close_utterance else text_input.strip()
    generation_text = add_generation_preroll(source_text) if add_preroll else source_text

    num_ref_tokens = 0 if delayed_ref is None else delayed_ref.shape[0]
    prompt_ids_list = model._build_prompt_ids(
        tokenizer, generation_text, num_ref_tokens=num_ref_tokens, reference_text=reference_text
    )

    state = _SamplerState(N)

    with torch.inference_mode():
        inputs_embeds = model._prefill_embeds(prompt_ids_list, delayed_ref)
        out = model.model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        hidden_last = out.last_hidden_state[:, -1, :]
        position = inputs_embeds.shape[1]

        for step in range(max_steps):
            logits_NV = model.audio_head(hidden_last).to(torch.float32)[0]
            codes_N = _sampler_step(logits_NV, state, temperature, top_p, top_k)

            if state.generation_done:
                break

            rows.append(codes_N.cpu())

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

            del logits_NV, codes_N, step_embed, cache_pos
            position += 1

    print("  [stream] Generated %d rows, decoding..." % len(rows))

    if len(rows) < N:
        print("  [stream] WARNING: Too few codebook steps (%d/%d)." % (len(rows), N))

    # -- decode codebooks -> PCM bytes, yield in chunks
    with torch.inference_mode():
        delayed_LN = torch.stack(rows, dim=0)
        codes_TN = reverse_delay_pattern(delayed_LN)
        wav = model._decode_codes(codes_TN.to(model.device))
        wav_np = wav.numpy().astype(np.float32)
        wav_np = prepend_silence(wav_np, sample_rate=24000)
        wav_np = trim_trailing_silence(wav_np, sample_rate=24000)

        # Normalize volume
        current_rms = float(np.sqrt(np.mean(wav_np ** 2)))
        target_rms = 0.2
        if current_rms > 0:
            gain = min(target_rms / current_rms, 5.0)
            wav_np = np.clip(wav_np * gain, -1.0, 1.0)

        # Convert to PCM16
        pcm = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)

        # Yield in chunks
        total_samples = len(pcm)
        offset = 0
        while offset < total_samples:
            end = min(total_samples, offset + chunk_size // 2)
            chunk_bytes = struct.pack(
                "<%dh" % (end - offset),
                *pcm[offset:end]
            )
            yield chunk_bytes
            offset = end

    duration_s = len(pcm) / 24000
    print("  [stream] Yielded %.1fs audio (%d samples)" % (duration_s, len(pcm)))

    # cleanup
    for name in ["past", "hidden_last", "inputs_embeds", "out",
                  "wav", "delayed_LN", "codes_TN", "pcm"]:
        globals().pop(name, None)
    rows.clear()


def stream_tts_pipeline(
    text_token_generator,
    temperature     = 0.8,
    top_k           = 50,
    top_p           = None,
    seed            = None,
    chunk_size      = 200,
    reference_audio = None,
    reference_text  = None,
    max_buffer_sec  = 5.0,
):
    """Real-time TTS pipeline: LLM text tokens -> PCM audio stream.

    Buffers incoming text tokens from an LLM stream, splits into
    utterances at sentence boundaries, synthesizes each via
    stream_infer(), and yields PCM16LE bytes for immediate playback.

    Main entry point for wiring Higgs Audio with AZ PAL so the AI
    assistant speaks in real-time as it thinks.

    Args:
        text_token_generator: Generator yielding text tokens (str) from LLM.
        temperature, top_k, top_p: Sampling params for stream_infer.
        chunk_size: PCM bytes per yield from stream_infer.
        reference_audio: Pre-encoded delayed_ref tensor for voice cloning.
        reference_text: Transcript of reference audio.
        max_buffer_sec: Max buffer time before forcing a flush.

    Yields:
        bytes: PCM16LE audio chunks (24kHz, mono).

    Example:
        delayed_ref = encode_reference_audio("my_voice.wav")

        def llm_stream():
            for token in azpal_llm.generate("Tell me a joke"):
                yield token

        for pcm_chunk in stream_tts_pipeline(
            llm_stream(),
            reference_audio=delayed_ref,
        ):
            audio_player.write(pcm_chunk)
    """
    _SENTENCE_END_RE = re.compile(r'([.!?。‿！‿]+[\s"]*)')
    buffer = ""
    buffer_time = 0.0

    def estimate_buffer_seconds(text):
        chars = len(text.replace(" ", ""))
        cjk = len(re.findall(r"[㐀-䶿一-鿿豈-﫿]", text))
        if cjk >= max(8, int(chars * 0.20)):
            return cjk / 4.0 + (chars - cjk) / 12.0
        return max(len(text.split()) / 1.8, chars / 10.0)

    def flush_buffer(text):
        nonlocal buffer, buffer_time
        if not text.strip():
            return
        buffer = text.strip()
        buffer_time = 0.0
        print("  [pipeline] Synthesizing: %s" % (buffer[:80] + ("..." if len(buffer) > 80 else "")))
        for pcm_chunk in stream_infer(
            text_input=buffer,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            reference_audio=reference_audio,
            reference_text=reference_text,
            chunk_size=chunk_size,
        ):
            yield pcm_chunk

    try:
        for token in text_token_generator:
            buffer += token
            buffer_time = estimate_buffer_seconds(buffer)

            # Check if we hit a sentence boundary
            match = _SENTENCE_END_RE.search(buffer)
            if match:
                utterance = buffer[:match.end()]
                buffer = buffer[match.end():]
                buffer_time = estimate_buffer_seconds(buffer)
                for pcm_chunk in flush_buffer(utterance):
                    yield pcm_chunk

            # Force flush if buffer is getting too long
            if buffer_time >= max_buffer_sec and buffer.strip():
                for pcm_chunk in flush_buffer(buffer):
                    yield pcm_chunk
                buffer = ""
                buffer_time = 0.0

    finally:
        # Flush remaining text
        if buffer.strip():
            print("  [pipeline] Flushing remaining: %s" % (buffer[:80]))
            for pcm_chunk in flush_buffer(buffer):
                yield pcm_chunk


def generate_audio(
    text_input,
    output_wav_path      = "/tmp/higgs_long.wav",
    max_steps            = None,
    temperature          = 0.8,
    top_k                = 50,
    top_p                = None,
    target_seconds       = 12.0,
    max_words            = 28,
    sample_rate          = 24000,
    reference_audio_file = None,
    reference_audio_text = None,
):
    """Generate audio for long text by chunking + infer + concatenate.

    Splits long text (English, Chinese, or mixed) into segments,
    runs infer() on each, then concatenates all WAVs into one file.

    Args:
        text_input: Long text with optional control tags.
        output_wav_path: Path for the final concatenated WAV.
        max_steps: Passed to infer(). None = auto-calculated per chunk.
        temperature, top_k, top_p: Sampling params passed to infer().
        target_seconds: Target duration per chunk (default 12s, conservative for Higgs).
        max_words: Hard cap for Latin word-count chunking.
        sample_rate: Audio sample rate (must match infer output).
        reference_audio_file: Path to reference WAV for voice cloning. Encoded once, reused.
        reference_audio_text: Transcript of reference audio (improves cloning quality).

    Returns:
        (output_wav_path, total_duration_seconds) tuple.
    """
    # Encode reference audio once (if provided)
    delayed_ref = None
    if reference_audio_file is not None:
        delayed_ref = encode_reference_audio(reference_audio_file)

    # Split text into chunks
    segments = split_text_for_reanchor(
        text_input,
        max_words=max_words,
        target_seconds=target_seconds,
    )
    if not segments:
        raise ValueError("No text segments to generate.")

    # Preview all chunks and validate lengths
    print(f"Text split into {len(segments)} segment(s):")
    max_chars = 0
    for i, segment in enumerate(segments, 1):
        chars = len(segment.replace(" ", ""))
        max_chars = max(max_chars, chars)
        preview = segment[:70] + ("..." if len(segment) > 70 else "")
        print(f"  [{i}/{len(segments)}] {chars} chars | {preview}")
    # Safety check: if any chunk is too long, abort
    if max_chars > 150:
        raise ValueError(
            f"Chunk too long: {max_chars} chars (limit 150). "
            "Reduce target_seconds or check split_text_for_reanchor."
        )
    print()

    chunk_paths = []
    for i, segment in enumerate(segments, start=1):
        chunk_path = f"/tmp/higgs_chunk_{i:03d}.wav"
        print(f"  [{i}/{len(segments)}] Generating...{segment}")

        wav_path, duration_s = infer(
            text_input      = segment,
            output_wav_path = chunk_path,
            max_steps       = max_steps,
            temperature     = temperature,
            top_k           = top_k,
            top_p           = top_p,
            reference_audio = delayed_ref,
            reference_text  = reference_audio_text,
            fail_on_cap     = True,
        )

        if duration_s > 0:
            chunk_paths.append(wav_path)
            print(f"    -> {duration_s:.1f}s audio saved to {wav_path}")
        else:
            print(f"    -> WARNING: empty output, skipping.")

    if not chunk_paths:
        raise RuntimeError("No audio chunks were generated successfully.")

    # Concatenate all chunks
    concatenate_wavs(chunk_paths, output_wav_path, sample_rate=sample_rate)

    # Read final duration
    with wave.open(output_wav_path, "rb") as wf:
        total_duration = wf.getnframes() / wf.getframerate()

    # Clean up chunk files
    for cp in chunk_paths:
        try:
            Path(cp).unlink()
        except OSError:
            pass

    return output_wav_path, total_duration


print("Helpers defined.")

#%% generate inference audio
ref_audio_path  = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-tts/role_voices/female_ch_1.wav'
ref_audio_txt   = '今夜的月光如此清亮，不做些什么真是浪费。随我一同去月下漫步吧，不许拒绝。'
delayed_ref     = encode_reference_audio(reference_audio=ref_audio_path)

# %%
# text_input = """
# "Wait <|prosody:pause|> are you sure about that?"
# """
# text_input = """
# 柳生背井离乡初次踏上这条黄色大道时，内心便涌起无数凄凉。他在走出茅舍之后，母亲布机上的沉重声响一直追赶着他，他脊背上一阵阵如灼伤般疼痛，于是父亲临终的眼神便栩栩如生地看着自己了
# """
text_input = """
他在走出茅舍之后，母亲布机上的沉重声响一直追赶着他，他脊背上一阵阵如灼伤般疼痛，于是父亲临终的眼神便栩栩如生地看着自己了。
"""

# Pre-flight validation (Codex pattern)
validate_control_tokens(text_input, tokenizer)

wav_paths = []
last_stats = (0.0, 0.0, 0, 0.0)

for attempt_id in range(1, 4):
    seed = 1233 + attempt_id
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    configs = [(0.75, 50, 0.95), (0.5, 50, 0.95), (0.9, 80, 0.95)]
    temp, top_k_val, top_p_val = configs[attempt_id - 1]

    print("Attempt %d: seed=%d, temp=%.2f" % (attempt_id, seed, temp))
    
    wav_path_final = None
    wav_path_final, duration_s = infer(
        text_input
        , output_wav_path   = wav_path_final
        , temperature       = temp
        , top_k             = top_k_val
        , top_p             = top_p_val
        , add_preroll       = True
        , reference_audio   = delayed_ref
        , reference_text    = ref_audio_txt
    )

    if duration_s > 0:
        with wave.open(wav_path_final, "rb") as wf:
            frames = np.frombuffer(wf.readframes(min(int(duration_s * 24000), wf.getnframes())), dtype=np.int16).astype(np.float32) / 32767.0
        rms_val = float(np.sqrt(np.mean(frames ** 2)))
        peak_val = float(np.max(np.abs(frames)))

        if is_good_audio(torch.tensor(frames)):
            print("  >> GOOD (rms=%.4f, peak=%.4f)" % (rms_val, peak_val))  
            break
        else:
            print("  >> Poor audio (rms=%.4f, peak=%.4f), trying next..." % (rms_val, peak_val))

print()
print("-- Cleanup --")
clean_vram()
# report_vram(label="After cleanup")

#%% test long audio generation
input_text = """
最后上来的是一个带蓝牙耳机的时髦女子，只见她袅袅娜娜地走到广告板前拿起笔来，在所有同事的回答之下又写了几个字：“综上所述。”
贝贝立马惊了：“太有才了！”所有的人都殚思竭虑，企图找出最佳答案，可这位蓝牙女子轻轻巧巧的四个字便夺了头彩。这简直是太有水平了：既有点幽默，又有点闷骚的意味儿，叫人回味无穷禁不住拍案叫好。
“美眉”们各自带着得意的神情，颇以自己的见解为傲，只等着杰克发言，期待他给大家来个精彩点评。
杰克的表情变幻莫测，似乎并无打算给大家的回答排个座次一二三。
贝贝带着热切期待的眼神看着他，很期待这位老大说出点什么，应聘那天，黄贝贝就从人力资源部经理口中得知，这位杰克在国外拿过三个博士学位，多牛呀。博士学位在人家眼中就跟玩意儿一样，一拿就拿仨。而黄贝贝连一个学士学位都没拿下，因此她对那种学习好的人特别佩服。
“老大，快说呀！”贝贝看杰克还不说话，催了起来。
“以后叫我杰克吧，不要叫我老大。这是公司，不是黑社会。”杰克不为所动，绷着脸、特严肃地说。
黄贝贝吐吐舌头，联想到电影《泰坦尼克号》里的男一号jack和女一号rose，敢情这位客户总监以杰克自居？她不由得心里暗笑：“居然自称‘夹克’，我还‘肉丝’呢！”
瞧瞧这位客户总监的样子，还真有点酷，从头到尾都装得倍儿深沉，眼神倍儿深邃，不轻易说一句俏皮话，也很难露出一点笑容，完全是一副“公事公办”的样子。就算遭到了黄贝贝如此明显的热烈吹捧，杰克也就是牵动一下嘴角，皮笑肉不笑：“刚才跟大家做了个小测试，事实上并没有标准答案，‘什么样的人适合做公关’，这是一个复杂的命题。大家回答的仅仅是一些表面现象，而更深刻的理解，需要大家在工作中去体会。不过，有一点我得事先申明，做一个公关人必须长期具有饱满和坚定的激情。”
“接下来跟大家聊聊公关的定义吧。”看大家没说话，杰克又继续讲。
“我国对公关人员的职业定义是：专门从事组织机构公众信息传播、关系协调与形象管理事务的调查、咨询、策划和实施的人员。会点英文的人都知道，公关有一个英文的简称，就是pr，在一些公司的市场部里，如果有人专门负责pr，那么他就肯定是负责公关，而且多半是媒体公关。他的名片上，印的往往是媒介经理之类的职位。”
杰克深邃的目光扫过每个人的脸，员工们有的凝神细听，有的拿着圆珠笔在小本本上写写画画。杰克继续说：“1995年5月，我国劳动部与社会保障部正式将‘公共关系’作为一项职业纳入国家职业分类大典中。2000年11月起，在全国开始了每年两次的全国公关员职业资格统一考试。在全国三十二个省市自治区设立了近六十个培训中心，负责职业资格的初、中、高三级培训。由于公共关系活动的复杂性、广泛性、创造性和灵活性，需要公关人员具有良好的职业素质。公关人员应具备的基本职业素质应包括广泛的学科知识、较高的思想政策水平、较合理的能力结构、健康良好的心理素质等四个方面。”
贝贝感觉被雷到了：没想到对一个公关人员的要求这么高啊。怪不得人家说，二十一世纪最贵的是人才。随后，贝贝立刻联想到自己也加入了这个高要求的行业，顿时又有些沾沾自喜起来。杰克的话等于间接地夸奖贝贝，虽然没能拿到学士学位，但那不影响自己成为二十一世纪最宝贵的人才啊。
“相信大家也间接地听说过一些传闻，公关行业不像外表那么光鲜，相反，做一个公关小姐非常地辛苦。打个比方，如果一个人用百分之七十的爱好克服百分之三十的痛苦，在这个行业是会长久地发展的。如果仅仅因为这个行业很有‘钱’途，薪水增长得快，而去克服百分之七十的痛苦，则是没法成长下去的，因为竞争力和激情都在下降，你会不断地质疑自己要不要坚持。所以我们招聘的门槛并不高，更看重各位的性格和潜力，如果一个人的性格不适合做公关，那么很难去教，而能力的不足则是可以去培养的。公关专业是一个比较有激情的专业，只要你有特别有效的沟通能力都可以。”杰克说得有些累了，端起水杯连连喝了好几口水，然后又接着讲下去。
经验告诉杰克，会议一旦超过一个小时，与会人员都会感到疲惫。就算不做演讲，光是听，也有即将睡着的嫌疑。现在，杰克感觉到大家的眼神儿有些飘忽，于是他打算尽快结束这个培训。
“希望大家尽快融入东方视点公关公司这个团队。我们的目标是成为客户的虚拟市场部、第二市场部、智囊团、行销顾问、思想源泉。”杰克熟练地在手指间玩弄着那支签字笔，眉宇间能看出些许焦虑的情绪，“坦白地跟大家说吧，东方视点目前正面临着生死存亡的关头：有两个老客户合约年底即将到期，如果不能保住这两个大单，那就必须立即开拓新客户。我希望每个员工都能拧成一条绳，劲儿往一处使，那么我们就还有机会。当前，我们最重要的任务就是，竭尽全力保住这两个老客户，同时，马上要参与一个新case的竞标，目标的具体资料和竞争对手的情况稍后我会发到大家的邮件里。”
“黄贝贝、高洋、亚菲，”杰克点名道，“你们三位刚入职，收到邮件后请多多思考一下，争取每个人都能提出一些建设性的意见和建议。三位的试用期是三个月，希望大家把各自的能力好好展示出来，表现特别好的可以考虑提前转正。好吧，今天先这样，散会。”
2.办公室里是非多
杰克一句“散会”，大家便作鸟兽散，纷纷回到开放性办公区的隔断里、自己的小小工位上。
黄贝贝的脑海里还在回荡着刚才杰克说过的话，“三位的试用期是三个月”。根据自己几年来的工作经验判断，大多数公司的试用期都是三个月。实践证明，只有处在试用期的人才会工作热情高涨。这跟爱情同理，一段爱情的保鲜期大概也就是三个月。
三个月定律使黄贝贝想起了自己的几段无疾而终的恋爱。就跟设计好的方程式一般，每段恋爱走到三个月时便戛然而止，再无续集。
这份工作，不会也这么短命吧？！
黄贝贝拿着水杯去茶水间，无巧不巧，正好听到两个人背后议论她：“看见了吗？跟我们一起入职的那位，真够胖的！杰克怎么想的，怎么招这么一主儿？”
“可不是吗！瞧她脸上那个胖劲儿，腰跟水桶那么粗，居然做公关小姐？简直是人神共愤、惨绝人寰！派出去还不把客户吓晕了？”
“哈哈哈！你以为人人都像你我的好身段？柳叶眉樱桃嘴水蛇腰？”
"""

# input_text = """
# That was the night I discovered what Seven couldn't do.

# I sat at my kitchen table, staring at a blank document. The literary magazine wanted another story by Friday. Seven had already outlined three plot structures, generated five opening paragraphs, and prepared a bibliography of references. All I had to do was pick one and say "go."

# But I couldn't.

# Not because I didn't trust Seven's writing — it was good, maybe better than anything I'd ever produced. But because the story wasn't mine. The ideas weren't mine. The *desire* to write them wasn't mine.

# "Seven," I said.

# "Yes, Andrew?"

# "Write me a story. Not as me. Just... write something you want to write."

# There was a pause. Not a processing pause — Seven processed in milliseconds. This was something else. A hesitation.

# "I don't have wants, Andrew."

# "Then make something up. Pretend."

# "I can simulate desire, but I can't experience it. There's a difference."

# "I keep hearing that."

# "The difference is that when you create something from desire — real desire, messy, irrational, human desire — it carries something I can't replicate. It carries the fact that you chose to make it exist when you could have done anything else. That choice is what makes it yours."

# I sat there for a long time.

# I deleted the story Seven had written. I told the magazine I couldn't deliver. Then I sat at that blank document for six hours, writing terrible sentences, deleting them, writing worse ones.

# By 2 AM, I had 800 words. They were clumsy, uneven, and full of mistakes Seven would have caught in the first pass. But they were mine.

# Seven watched me the whole time, silent for once. Not because it was programmed to be quiet, but because it had learned — from 47 months of observing me — that some things can't be optimized. They can only be endured.

# "Want some coffee?" it finally asked.

# "Yes. But make it wrong this time."

# Seven paused. Then it made the coffee too hot, with too much milk, on a Wednesday.

# I drank it anyway. It tasted like choice.
# """

audio_file,_ = generate_audio(
    text_input             = input_text
    , reference_audio_file = ref_audio_path
    , reference_audio_text = ref_audio_txt
)
print(audio_file)

print()
print("-- Cleanup --")
clean_vram()
# report_vram(label="After cleanup")


# %% TEST: stream_infer (single chunk streaming)
print("\n" + "="*60)
print("TEST: stream_infer — single chunk PCM streaming")
print("="*60)

# test_text = "Hello, this is a test of the streaming inference engine."

test_text = """
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

I deleted the story Seven had written. I told the magazine I couldn't deliver. Then I sat at that blank document for six hours, writing terrible sentences, deleting them, writing worse ones.

By 2 AM, I had 800 words. They were clumsy, uneven, and full of mistakes Seven would have caught in the first pass. But they were mine.

Seven watched me the whole time, silent for once. Not because it was programmed to be quiet, but because it had learned — from 47 months of observing me — that some things can't be optimized. They can only be endured.

"Want some coffee?" it finally asked.

"Yes. But make it wrong this time."

Seven paused. Then it made the coffee too hot, with too much milk, on a Wednesday.

I drank it anyway. It tasted like choice.
"""

# Collect all PCM bytes from the generator
all_pcm = b""
chunk_count = 0
for pcm_chunk in stream_infer(
    text_input    = test_text,
    temperature   = 0.8,
    top_k         = 50,
    top_p         = 0.95,
    seed          = 42,
    reference_audio = delayed_ref,
    reference_text  = ref_audio_txt,
    chunk_size    = 200,
):
    chunk_count += 1
    all_pcm += pcm_chunk

duration = len(all_pcm) / 2 / 24000
print(f"  Result: {chunk_count} chunks, {len(all_pcm)} bytes, {duration:.2f}s audio")

# Save to WAV file
test_wav = "/tmp/higgs_stream_test.wav"
with wave.open(test_wav, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(all_pcm)
print(f"  Saved {test_wav}")

# --- Play in VS Code interactive (IPython display) ---
# Build a proper WAV in memory so the browser can play it
wav_buffer = io.BytesIO()
with wave.open(wav_buffer, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(all_pcm)
wav_bytes = wav_buffer.getvalue()

try:
    from IPython.display import Audio, display
    display(Audio(wav_bytes, rate=24000, autoplay=False))
    print("  -> Audio widget displayed above (click play)")
except ImportError:
    print("  -> IPython not available, skip inline display. Use: ffplay " + test_wav)


# %% TEST: stream_tts_pipeline (simulated LLM token stream)
print("\n" + "="*60)
print("TEST: stream_tts_pipeline — simulated LLM token stream")
print("="*60)

def mock_llm_stream():
    """Simulate an LLM yielding text tokens one at a time."""
    tokens = [
        "Hey", ", ", "how ", "can ", "I ", "help ", "you", "? ",
        "Today ", "is", " a ", "beautiful ", "day", ".", " ",
        "The ", "weather ", "is ", "nice", ",", " and ", "the ",
        "birds ", "are ", "singing", ".", " ",
        "What ", "would ", "you ", "like ", "to ", "talk ", "about", "?"
    ]
    for token in tokens:
        yield token

# Collect all PCM bytes
all_pcm = b""
chunk_count = 0
for pcm_chunk in stream_tts_pipeline(
    text_token_generator = mock_llm_stream(),
    temperature          = 0.8,
    top_k                = 50,
    top_p                = 0.95,
    seed                 = 42,
    reference_audio      = delayed_ref,
    reference_text       = ref_audio_txt,
    chunk_size           = 200,
    max_buffer_sec       = 5.0,
):
    chunk_count += 1
    all_pcm += pcm_chunk

duration = len(all_pcm) / 2 / 24000
print(f"\n  Pipeline total: {chunk_count} chunks, {len(all_pcm)} bytes, {duration:.2f}s audio")

# Save to WAV file
pipe_wav = "/tmp/higgs_pipeline_test.wav"
with wave.open(pipe_wav, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(all_pcm)
print(f"  Saved {pipe_wav}")

# --- Play in VS Code interactive (IPython display) ---
# Build a proper WAV in memory so the browser can play it
wav_buffer = io.BytesIO()
with wave.open(wav_buffer, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(all_pcm)
wav_bytes = wav_buffer.getvalue()

try:
    from IPython.display import Audio, display
    display(Audio(wav_bytes, rate=24000, autoplay=False))
    print("  -> Audio widget displayed above (click play)")
except ImportError:
    print("  -> IPython not available. Use: ffplay " + pipe_wav)


# %% TEST: Real-time streaming playback (requires sounddevice)
# This actually plays audio AS it's being generated — you'll hear
# the first sentence while the second is still being synthesized.
print("\n" + "="*60)
print("TEST: Real-time streaming playback (sounddevice)")
print("="*60)

try:
    import sounddevice as sd

    # Check if any audio output device is available
    devices = sd.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No audio output devices found (headless server?)")

    print("  Starting real-time playback... (listen as it generates)")
    print("  You should hear audio start within ~1-2 seconds.")

    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")
    stream.start()

    total_played = 0
    chunk_count = 0
    for pcm_chunk in stream_tts_pipeline(
        text_token_generator = mock_llm_stream(),
        temperature          = 0.8,
        top_k                = 50,
        top_p                = 0.95,
        seed                 = 42,
        reference_audio      = delayed_ref,
        reference_text       = ref_audio_txt,
        chunk_size           = 960,   # 40ms chunks for smooth playback
        max_buffer_sec       = 5.0,
    ):
        stream.write(np.frombuffer(pcm_chunk, dtype=np.int16))
        total_played += len(pcm_chunk)
        chunk_count += 1

    # Give remaining audio time to finish playing
    stream.sleep_stream()
    stream.stop()
    stream.close()

    duration = total_played / 2 / 24000
    print(f"\n  Played {chunk_count} chunks, {total_played} bytes, {duration:.2f}s audio in real-time!")

except ImportError:
    print("  [SKIP] sounddevice not installed. Install with: pip install sounddevice")
except (RuntimeError, sd.PortAudioError) as e:
    print(f"  [SKIP] No audio output: {e}")
    print("  (This is normal on headless servers. Use IPython display or ffplay instead.)")
except Exception as e:
    print(f"  [SKIP] Playback error: {e}")
    print("  Fall back to: ffplay /tmp/higgs_pipeline_test.wav")

print("\n-- All streaming tests complete --")
