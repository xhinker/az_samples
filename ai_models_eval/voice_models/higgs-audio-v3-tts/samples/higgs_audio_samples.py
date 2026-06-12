#!/usr/bin/env python3
"""Higgs Audio v3 TTS — Fixed Inference with Codex patterns."""

#%% [markdown]
# # Higgs Audio v3 TTS — Improved Control Tag Handling

#%%
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import re
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

def prepend_silence(wav_np, sample_rate=24000, silence_sec=POST_DECODE_SILENCE_SEC):
    """Prepend a short playback guard.

    This is only for players that clip the first samples. It cannot recover a
    missing phoneme because it runs after the neural codec has already decoded.
    """
    silence_samples = int(silence_sec * sample_rate)
    silence = np.zeros(silence_samples, dtype=wav_np.dtype)
    return np.concatenate([silence, wav_np])

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
    top_p           = 0.9,
    seed            = None,
    add_preroll     = True,
):
    """Run full AR generation + vocoder decode for *text_input*.

    Args:
        text_input: Text with optional control tags (emotion, prosody, style, sfx).
        output_wav_path: Path to save the output WAV (24kHz, 16-bit PCM). None = auto unique path.
        max_steps: Hard cap on AR generation steps. If None (default), auto-calculated
                   from text length: min(2048, max(192, max(words*4, chars*2) + 160)).
                   Each step ~1 audio frame; 1024 steps ~= 13s of audio.
                   EOC token naturally stops generation early, so this is a safety cap.
        temperature: Sampling temperature (0.5-1.2). Lower = deterministic, higher = creative.
        top_k: Keep only top-K tokens before sampling. None = disabled.
        top_p: Nucleus sampling threshold. None = disabled.
        seed: Random seed for reproducible generation. None = random each time.

    Returns:
        (wav_path, duration_seconds) tuple.

    Uses official multi-codebook delay sampler with proper BOC/EOC handling.
    """

    device_idx = int(str(model.device).split(":")[-1])
    N = model.num_codebooks

    # -- dynamic max_steps: use both word count (English) and char count (CJK), clamped [192, 2048]
    if max_steps is None:
        words = len(text_input.split())
        chars = len(text_input.replace(" ", ""))
        max_steps = min(2048, max(512, max(int(words * 4), int(chars * 2)) + 160))

    # -- auto unique output path if not specified
    if output_wav_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_wav_path = f"/tmp/higgs_{ts}_{id(text_input):08x}.wav"

    # -- set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -- build prompt embeddings ---------------------------------
    rows = []
    report_vram(device_idx, "Before AR generation")

    generation_text = add_generation_preroll(text_input) if add_preroll else text_input.strip()
    if generation_text != text_input.strip():
        print("  Added model-level preroll before first spoken token.")

    prompt_ids_list = model._build_prompt_ids(
        tokenizer, generation_text, num_ref_tokens=0, reference_text=None
    )

    state = _SamplerState(N)

    with torch.inference_mode():
        inputs_embeds   = model._prefill_embeds(prompt_ids_list, None)
        out             = model.model(inputs_embeds=inputs_embeds, use_cache=True)
        past            = out.past_key_values
        hidden_last     = out.last_hidden_state[:, -1, :]
        position        = inputs_embeds.shape[1]

        for step in range(max_steps):
            logits_NV = model.audio_head(hidden_last).to(torch.float32)[0]  # [N, V]
            codes_N = _sampler_step(logits_NV, state, temperature, top_p, top_k)

            if step == 0:
                print("Sampling config: temperature=%.1f, top_k=%d, top_p=%.1f" % (temperature, top_k if top_k else 0, top_p if top_p else 0))

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

    report_vram(device_idx, "After AR generation")
    print("Generated %d rows (codebooks), max_steps was %d" % (len(rows), max_steps))
    if len(rows) >= max_steps - N:
        print("  WARNING: hit max_steps cap! Text may be truncated. Increase max_steps or use generate_audio() for long text.")

    if len(rows) < N:
        print("WARNING: Too few codebook steps (%d/%d). Output may be silent." % (len(rows), N))

    # -- decode codebooks -> WAV ---------------------------------
    with torch.inference_mode():
        delayed_LN = torch.stack(rows, dim=0)
        codes_TN = reverse_delay_pattern(delayed_LN)
        wav = model._decode_codes(codes_TN.to(model.device))
        wav_np = wav.numpy().astype(np.float32)
        wav_np = prepend_silence(wav_np, sample_rate=24000)
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
    
    report_vram(device_idx, "After decode")

    # cleanup intermediates (codec cleared separately below)
    for name in [
        "past", "hidden_last", "inputs_embeds", "out",
        "logits_NV", "codes_N", "step_embed", "cache_pos",
        "wav", "delayed_LN", "codes_TN", "pcm",
    ]:
        globals().pop(name, None)

    rows.clear()

    return output_wav_path, duration_s

def generate_audio(
    text_input,
    output_wav_path="/tmp/higgs_long.wav",
    max_steps=None,
    temperature=0.65,
    top_k=50,
    top_p=0.9,
    target_seconds=12.0,
    sample_rate=24000,
):
    """Generate audio for long text by chunking + infer + concatenate.

    Splits long text (English, Chinese, or mixed) into segments,
    runs infer() on each, then concatenates all WAVs into one file.

    Args:
        text_input: Long text with optional control tags.
        output_wav_path: Path for the final concatenated WAV.
        max_steps: Passed to infer(). None = auto-calculated per chunk.
        temperature, top_k, top_p: Sampling params passed to infer().
        target_seconds: Target duration per chunk (default 12s, well under 27s max).
        sample_rate: Audio sample rate (must match infer output).

    Returns:
        (output_wav_path, total_duration_seconds) tuple.
    """
    # Split text into chunks
    segments = split_text_for_reanchor(text_input, target_seconds=target_seconds)
    print(segments)
    if not segments:
        raise ValueError("No text segments to generate.")

    print(f"Text split into {len(segments)} segment(s) for generation.")

    chunk_paths = []
    for i, segment in enumerate(segments, start=1):
        chunk_path = f"/tmp/higgs_chunk_{i:03d}.wav"
        print(f"  [{i}/{len(segments)}] Generating: {segment[:80]}{'...' if len(segment) > 80 else ''}")

        wav_path, duration_s = infer(
            text_input      = segment,
            output_wav_path = chunk_path,
            max_steps       = max_steps,
            temperature     = temperature,
            top_k           = top_k,
            top_p           = top_p,
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

# %% [markdown]
# ## 4. Run inference then clean up VRAM

# %%
# text_input = """
# "Wait <|prosody:pause|> are you sure about that?"
# """
# text_input = """
# 数日前，柳生背井离乡初次踏上这条黄色大道时，内心便涌起无数凄凉。他在走出茅舍之后，母亲布机上的沉重声响一直追赶着他，他脊背上一阵阵如灼伤般疼痛，于是父亲临终的眼神便栩栩如生地看着自己了
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
        , output_wav_path=wav_path_final
        , temperature=temp
        , top_k=top_k_val
        , top_p=top_p_val
        , add_preroll = True
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
report_vram(label="After cleanup")

#%% test long audio generation
input_text = """
此后再不曾在道上遇上往来之人。
数日前，柳生背井离乡初次踏上这条黄色大道时，内心便涌起无数凄凉。他在走出茅舍之后，母亲布机上的沉重声响一直追赶着他，他脊背上一阵阵如灼伤般疼痛，于是父亲临终的眼神便栩栩如生地看着自己了。为了光耀祖宗，他踏上了黄色大道。姹紫嫣红的春天景色如一卷画一般铺展开来，柳生却视而不见。展现在他眼前的仿佛是一派暮秋落叶纷扬，足下的黄色大道也显得虚无缥缈。
"""

audio_file,_ = generate_audio(text_input=input_text)
print(audio_file)

print()
print("-- Cleanup --")
clean_vram()
report_vram(label="After cleanup")
