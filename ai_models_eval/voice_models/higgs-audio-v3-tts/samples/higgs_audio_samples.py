# %% [markdown]
# # Higgs Audio v3 TTS — Fixed Inference (Sampling + Proper EOC)

# %%
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
    trust_remote_code = True,
    dtype             = torch.bfloat16,
    device_map        = "cuda:1",
)
model.eval()
model.requires_grad_(False)

print("Model loaded on:", model.device)
print("Num codebooks:", model.num_codebooks)

# ## 3. Helper Functions (run this cell first!)
BOC_ID = 1024
EOC_ID = 1025

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
    free_gb = torch.cuda.mem_get_info(device_idx)[0] / 1e9
    total_gb = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
    allocated_mb = torch.cuda.memory_allocated(device_idx) / 1e6
    reserved_mb = torch.cuda.memory_reserved(device_idx) / 1e6
    max_alloc_mb = torch.cuda.max_memory_allocated(device_idx) / 1e6

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
        # Delay pattern: fill remaining codebooks with BOC placeholders
        next_cb = state.delay_count + 1
        if next_cb < N:
            codes_N[next_cb:] = BOC_ID
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_N[0].item()) == EOC_ID:
        # Start eoc countdown (N-2 more steps to drain remaining codebooks)
        if N <= 2:
            state.generation_done = True
        else:
            state.eoc_countdown = N - 2

    return codes_N


def infer(
    text_input,
    output_wav_path="/tmp/higgs_test.wav",
    max_steps=2048,
    temperature=0.8,
    top_k=50,
    top_p=None,
):
    """Run full AR generation + vocoder decode for *text_input*.

    Returns (wav_path, duration_seconds).

    Uses official sampling logic with proper delay pattern and EOC countdown to fix
    blank/broken audio on longer texts with control tokens.
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
text_input = (
    "<|emotion:determination|><|prosody:expressive_high|>"
    "Higgs Audio v3 TTS is built for voice chat: it speaks, not just reads. It turns model responses into expressive conversational speech across 100+ languages"
    "<|sfx:laughter|>haha, this so cool, so great"
)

wav_path, duration_s = infer(
    text_input,
    temperature=0.8,   # official default (NOT 1.0/argmax!)
    top_k=50,           # recommended from SGLang-Omni cookbook
)

print()
print("-- Cleanup --")
clean_vram()
report_vram(label="After cleanup")
