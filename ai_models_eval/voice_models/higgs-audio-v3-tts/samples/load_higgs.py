# %% [markdown]
# # Higgs Audio v3 TTS — Minimal Load & Test

# %%
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b"

# %% [markdown]
# ## 1. Load Tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Tokenizer loaded.")

# %% [markdown]
# ## 2. Load Model (bfloat16, direct GPU)

# %%
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda:1",
).eval()
model.requires_grad_(False)

print("Model loaded on:", model.device)
print("Num codebooks:", model.num_codebooks)

# %% [markdown]
# ## 3. Helper Functions (run this cell first!)

# %%
BOC_ID = 1024
EOC_ID = 1025

def reverse_delay_pattern(delayed_LN):
    L, Nc = delayed_LN.shape
    T = L - (Nc - 1)
    out = torch.empty((T, Nc), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(Nc):
        out[:, c] = delayed_LN[c : c + T, c]
    return out

print("Helpers defined.")

# %% [markdown]
# ## 4. Generate & Decode with VRAM diagnostics

# %%
import numpy as np
import wave
import gc

device_idx = int(str(model.device).split(':')[-1])

def report_vram(label=""):
    torch.cuda.synchronize(device_idx)
    free_gb = torch.cuda.mem_get_info(device_idx)[0] / 1e9
    total_gb = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
    allocated_mb = torch.cuda.memory_allocated(device_idx) / 1e6
    reserved_mb = torch.cuda.memory_reserved(device_idx) / 1e6
    max_alloc_mb = torch.cuda.max_memory_allocated(device_idx) / 1e6
    print("[%s] Free: %.1f/%.0f GB | Allocated: %.0f MB | Reserved pool: %.0f MB | Peak ever: %.0f MB" 
          % (label, free_gb, total_gb, allocated_mb, reserved_mb, max_alloc_mb))

report_vram("After model load")

# text_input = "The issue is likely that PyTorch's caching allocator keeps freed memory pooled — it doesn't return it to the system unless forced."
text_input = "The issue is likely that PyTorch's caching allocator keeps freed memory pooled"
N = model.num_codebooks

prompt_ids_list = model._build_prompt_ids(
    tokenizer, text_input, num_ref_tokens=0, reference_text=None
)

max_steps = min(2048, max(192, int(len(text_input.split()) * 4) + 160))
rows = []

with torch.inference_mode():
    inputs_embeds = model._prefill_embeds(prompt_ids_list, None)
    out = model.model(inputs_embeds=inputs_embeds, use_cache=True)
    past = out.past_key_values
    hidden_last = out.last_hidden_state[:, -1, :]
    position = inputs_embeds.shape[1]

    for step in range(max_steps):
        logits_NV = model.audio_head(hidden_last).to(torch.float32)[0]
        codes_N = logits_NV.argmax(dim=-1).to(torch.long)
        rows.append(codes_N.cpu())

        if int(codes_N[0].item()) == EOC_ID and step > N:
            break

        step_embed = model.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
        cache_pos = torch.tensor([position], device=model.device)
        out = model.model(
            inputs_embeds=step_embed.to(inputs_embeds.dtype),
            past_key_values=past, use_cache=True, cache_position=cache_pos,
        )
        past = out.past_key_values
        hidden_last = out.last_hidden_state[:, -1, :]
        del logits_NV, codes_N, step_embed, cache_pos
        position += 1

report_vram("After AR generation")
print("Generated", len(rows), "rows")

# Decode to WAV
with torch.inference_mode():
    delayed_LN = torch.stack(rows, dim=0)
    codes_TN = reverse_delay_pattern(delayed_LN)
    wav = model._decode_codes(codes_TN.to(model.device))
    wav_np = wav.numpy().astype(np.float32)

with wave.open("/tmp/higgs_test.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    pcm = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
    wf.writeframes(pcm.tobytes())

print("Saved /tmp/higgs_test.wav (%.1fs audio)" % (len(wav_np)/24000))
report_vram("After decode")

# ── Full VRAM cleanup ───────────────────────────────────────────
print("\nCleaning up...")

for var_name in [
    'past', 'hidden_last', 'inputs_embeds', 'out',
    'logits_NV', 'codes_N', 'step_embed', 'cache_pos',
    'wav', 'delayed_LN', 'codes_TN', 'pcm', 'wav_np',
]:
    obj = globals().pop(var_name, None)
    if hasattr(obj, 'device') and str(getattr(obj, 'device', '')).startswith('cuda'):
        obj.cpu()
    del obj

rows.clear()
globals().pop('rows', None)

# Clear codec cache (DAC vocoder model stays on GPU after decode)
if hasattr(model, '_audio_codec') and model._audio_codec is not None:
    print("Clearing cached audio codec from GPU...")
    model._audio_codec.cpu()
    model._audio_codec = None

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

report_vram("After cleanup")

# ── Aggressive cleanup: try to force allocator to release memory ──
print("\nAttempting aggressive cleanup...")
torch.cuda.reset_peak_memory_stats(device_idx)
gc.collect()
torch.cuda.empty_cache()

report_vram("After aggressive cleanup")

print("""
NOTE: If 'Allocated' stays high (~16GB), it's likely PyTorch's caching allocator.

SOLUTION: Set this env var BEFORE starting Python to enable segment shrinking:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

This lets the allocator shrink freed segments instead of pooling them forever.
Without it, PyTorch keeps peak memory pooled for reuse (faster but wastes VRAM).
""")
