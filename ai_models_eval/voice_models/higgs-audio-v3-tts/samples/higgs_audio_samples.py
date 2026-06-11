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
    if peak > 0.99:  # clipping
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
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_idx, remove)
        logits = torch.where(scatter, float("-inf"), logits)

    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1)


class _SamplerState:
    """Official sampler state machine (from modeling code)."""
    def __init__(self, num_codebooks):
        self.num_codebooks = num_codebooks
        self.delay_count = 0
        self.eoc_countdown = None
        self.generation_done = False


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
    output_wav_path="/tmp/higgs_test.wav",
    max_steps=1024,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):
    """Run full AR generation + vocoder decode for *text_input*.

    Returns (wav_path, duration_seconds).

    Uses official sampling logic with proper delay pattern and EOC countdown.
    Also leverages model.generate_speech() for best control tag following.
    """
    import numpy as np
    import wave

    device_idx = int(str(model.device).split(":")[-1])
    N = model.num_codebooks

    # -- build prompt embeddings ---------------------------------
    rows = []
    report_vram(device_idx, "Before AR generation")

    prompt_ids_list = model._build_prompt_ids(
        tokenizer, text_input, num_ref_tokens=0, reference_text=None
    )

    state = _SamplerState(N)

    with torch.inference_mode():
        inputs_embeds = model._prefill_embeds(prompt_ids_list, None)
        out = model.model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        hidden_last = out.last_hidden_state[:, -1, :]
        position = inputs_embeds.shape[1]

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
    print("Generated %d rows (codebooks)" % len(rows))

    if len(rows) < N:
        print("WARNING: Too few codebook steps (%d/%d). Output may be silent." % (len(rows), N))

    # -- decode codebooks -> WAV ---------------------------------
    with torch.inference_mode():
        delayed_LN = torch.stack(rows, dim=0)
        codes_TN = reverse_delay_pattern(delayed_LN)
        wav = model._decode_codes(codes_TN.to(model.device))
        wav_np = wav.numpy().astype(np.float32)

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


print("Helpers defined.")

# %% [markdown]
# ## 4. Run inference then clean up VRAM

# %%
# text_input = (
#     "<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. "
#     "<|sfx:laughter|>Hehe, no, seriously, I was not ready for that."
# )

# text_input = (
#     "<|emotion:disgust|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. "
#     "<|emotion:fear|>Hehe, no, seriously, I was not ready for that."
# )

text_input = (
    "<|emotion:sadness|>hey, how can I help you today? same voice, same words, and uh, a completely different presence!"
)

text_input = """
<|emotion:sadness|>Hi, I am so sad today, I wanno cry, could you hug me!
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

    configs = [(0.8, None, None), (0.75, 50, 0.95), (0.9, 80, 0.95)]
    temp, top_k_val, top_p_val = configs[attempt_id - 1]

    print("Attempt %d: seed=%d, temp=%.2f" % (attempt_id, seed, temp))
    
    wav_path_final = "/tmp/higgs_seed%d.wav" % seed
    _, duration_s = infer(text_input, output_wav_path=wav_path_final, temperature=temp, top_k=top_k_val, top_p=top_p_val)

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
