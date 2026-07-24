"""
llm_inference.py — Minimal LLM Inference with MLX
==================================================
Loads real Qwen3-0.6B weights and generates text.
No PyTorch, no transformers library — just MLX arrays and math.

Requirements: pip install mlx tokenizers
Run:          python llm_inference.py
"""

import json
import time
import mlx.core as mx
from tokenizers import Tokenizer

# ═══════════════════════════════════════════════════════════════════════
# Config — loaded from the model's config.json
# ═══════════════════════════════════════════════════════════════════════
MODEL_DIR = "Qwen3-0.6B"

with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)

D_MODEL  = cfg["hidden_size"]           # 1024
N_LAYERS = cfg["num_hidden_layers"]      # 28
N_HEADS  = cfg["num_attention_heads"]    # 16
N_KV     = cfg["num_key_value_heads"]    # 8  (GQA: 2 Q heads share 1 KV head)
D_HEAD   = cfg["head_dim"]              # 128
D_FF     = cfg["intermediate_size"]      # 3072
VOCAB    = cfg["vocab_size"]            # 151,936
EPS      = cfg["rms_norm_eps"]          # 1e-6
THETA    = cfg["rope_theta"]            # 1,000,000
EOS_ID   = cfg["eos_token_id"]          # 151645

# ═══════════════════════════════════════════════════════════════════════
# Load weights & tokenizer
# ═══════════════════════════════════════════════════════════════════════
W   = mx.load(f"{MODEL_DIR}/model.safetensors")
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")


# ═══════════════════════════════════════════════════════════════════════
# 1. RMSNorm — "scale by magnitude, then by learned weights"
# ═══════════════════════════════════════════════════════════════════════
def rms_norm(x, w):
    """RMSNorm: x / sqrt(mean(x²) + ε) * w"""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS) * w


# ═══════════════════════════════════════════════════════════════════════
# 2. RoPE — "weave position into vectors through rotation"
# ═══════════════════════════════════════════════════════════════════════
def apply_rope(q, k, positions):
    """
    Rotary Position Embedding.
    q: (n_heads, seq, d_head)
    k: (n_kv_heads, seq, d_head)
    positions: list[int] like [0, 1, 2, ...]
    """
    half = D_HEAD // 2  # 64
    # Frequency for each dimension pair: θ_i = 1 / THETA^(i / half)
    freqs = 1.0 / (THETA ** (mx.arange(half, dtype=mx.float32) / half))
    # Angle = position × frequency  →  (seq, half)
    angles = mx.array(positions, dtype=mx.float32)[:, None] * freqs[None, :]
    cos = mx.cos(angles)[None, :, :]  # (1, seq, half) — broadcast over heads
    sin = mx.sin(angles)[None, :, :]

    def rotate(x):
        x1 = x[..., :half]    # first half
        x2 = x[..., half:]    # second half
        # Rotate each pair: [x1, x2] → [x1·cos − x2·sin, x1·sin + x2·cos]
        return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    return rotate(q), rotate(k)


# ═══════════════════════════════════════════════════════════════════════
# 3. Grouped Query Attention with KV Cache
# ═══════════════════════════════════════════════════════════════════════
def attention(x, layer_idx, cache, positions):
    """
    GQA attention with KV cache.
    x:         (seq, 1024)  — input hidden states
    cache:     list of (k, v) or None per layer
    positions: list[int]
    Returns:   (seq, 1024)
    """
    p = f"model.layers.{layer_idx}.self_attn"

    # --- Linear projections: x @ W.T ---
    q = x @ W[f"{p}.q_proj.weight"].T   # (seq, 2048) = 16 heads × 128
    k = x @ W[f"{p}.k_proj.weight"].T   # (seq, 1024) =  8 heads × 128
    v = x @ W[f"{p}.v_proj.weight"].T   # (seq, 1024) =  8 heads × 128

    # --- Split into heads ---
    q = q.reshape(-1, N_HEADS, D_HEAD).transpose(1, 0, 2)    # (16, seq, 128)
    k = k.reshape(-1, N_KV, D_HEAD).transpose(1, 0, 2)       # (8, seq, 128)
    v = v.reshape(-1, N_KV, D_HEAD).transpose(1, 0, 2)       # (8, seq, 128)

    # --- QK-Norm (Qwen3-specific: RMSNorm on each head before RoPE) ---
    q = rms_norm(q, W[f"{p}.q_norm.weight"])
    k = rms_norm(k, W[f"{p}.k_norm.weight"])

    # --- Apply RoPE ---
    q, k = apply_rope(q, k, positions)

    # --- KV Cache: append new keys/values ---
    if cache[layer_idx] is not None:
        ck, cv = cache[layer_idx]
        k = mx.concatenate([ck, k], axis=1)   # (8, total, 128)
        v = mx.concatenate([cv, v], axis=1)
    cache[layer_idx] = (k, v)

    # --- GQA: expand 8 KV heads → 16 (each KV head shared by 2 Q heads) ---
    rep = N_HEADS // N_KV  # 2
    k = mx.repeat(k, rep, axis=0)   # (16, total, 128)
    v = mx.repeat(v, rep, axis=0)

    # --- Scaled dot-product attention ---
    scale = D_HEAD ** -0.5
    scores = (q @ k.transpose(0, 2, 1)) * scale   # (16, seq, total)

    # --- Causal mask (only needed during prefill, not decode) ---
    seq_len = q.shape[1]
    if seq_len > 1:
        total = k.shape[1]
        # Add -inf above the diagonal so tokens can't see the future
        mask = mx.triu(mx.full((seq_len, total), -1e9, dtype=scores.dtype), k=1)
        scores = scores + mask

    # --- Softmax + weighted sum of V ---
    attn = mx.softmax(scores, axis=-1)
    out  = attn @ v                                  # (16, seq, 128)

    # --- Merge heads + output projection ---
    out = out.transpose(1, 0, 2).reshape(-1, N_HEADS * D_HEAD)  # (seq, 2048)
    return out @ W[f"{p}.o_proj.weight"].T                        # (seq, 1024)


# ═══════════════════════════════════════════════════════════════════════
# 4. SwiGLU Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════
def ffn(x, layer_idx):
    """SwiGLU: (silu(x·W_gate) * (x·W_up)) · W_down"""
    p = f"model.layers.{layer_idx}.mlp"
    gate = x @ W[f"{p}.gate_proj.weight"].T   # (seq, 3072)
    up   = x @ W[f"{p}.up_proj.weight"].T     # (seq, 3072)
    silu = gate * mx.sigmoid(gate)             # SiLU(x) = x · σ(x)
    return (silu * up) @ W[f"{p}.down_proj.weight"].T  # (seq, 1024)


# ═══════════════════════════════════════════════════════════════════════
# 5. Transformer Block — "norm → attn → residual → norm → ffn → residual"
# ═══════════════════════════════════════════════════════════════════════
def transformer_block(x, layer_idx, cache, positions):
    p = f"model.layers.{layer_idx}"
    # Pre-norm attention + residual
    h = rms_norm(x, W[f"{p}.input_layernorm.weight"])
    x = x + attention(h, layer_idx, cache, positions)
    # Pre-norm FFN + residual
    h = rms_norm(x, W[f"{p}.post_attention_layernorm.weight"])
    x = x + ffn(h, layer_idx)
    return x


# ═══════════════════════════════════════════════════════════════════════
# 6. Forward Pass — embed → blocks → norm → LM head
# ═══════════════════════════════════════════════════════════════════════
def forward(token_ids, cache, start_pos):
    positions = list(range(start_pos, start_pos + len(token_ids)))

    # Embedding lookup: just index into the weight matrix
    x = W["model.embed_tokens.weight"][mx.array(token_ids)]   # (seq, 1024)

    # Run through all transformer blocks
    for i in range(N_LAYERS):
        x = transformer_block(x, i, cache, positions)

    # Final norm + project to vocabulary
    x = rms_norm(x, W["model.norm.weight"])
    logits = x @ W["lm_head.weight"].T   # (seq, 151936)
    return logits


# ═══════════════════════════════════════════════════════════════════════
# 7. Sampling — turn logits into a token ID
# ═══════════════════════════════════════════════════════════════════════
def sample(logits, temperature=0.0, top_p=0.95):
    """Greedy (temp=0) or nucleus sampling."""
    if temperature == 0:
        return int(mx.argmax(logits))

    logits = logits / temperature
    probs  = mx.softmax(logits)

    # Top-p: keep smallest set of tokens whose cumulative prob ≥ top_p
    order    = mx.argsort(-probs)
    sorted_p = probs[order]
    cumsum   = mx.cumsum(sorted_p)
    cutoff   = int(mx.sum(cumsum < top_p)) + 1

    sorted_p = sorted_p[:cutoff]
    sorted_p = sorted_p / mx.sum(sorted_p)
    order    = order[:cutoff]

    idx = mx.random.categorical(mx.log(sorted_p[None, :]))
    return int(order[idx])


# ═══════════════════════════════════════════════════════════════════════
# 8. Generate — the main loop
# ═══════════════════════════════════════════════════════════════════════
def generate(prompt, max_tokens=100, temperature=0.0, top_p=0.95):
    # Tokenize
    ids = tok.encode(prompt).ids

    # --- Prefill: process all input tokens at once ---
    cache  = [None] * N_LAYERS
    logits = forward(ids, cache, 0)
    next_id = sample(logits[-1], temperature, top_p)
    ids.append(next_id)

    # --- Decode: generate one token at a time ---
    for _ in range(max_tokens - 1):
        if next_id == EOS_ID:
            break
        pos      = len(ids) - 1
        logits   = forward([next_id], cache, pos)
        next_id  = sample(logits[-1], temperature, top_p)
        ids.append(next_id)

    return tok.decode(ids)


# ═══════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    prompt = "The capital of France is"

    print(f"Model:  Qwen3-0.6B  ({N_LAYERS} layers, {D_MODEL} dim, "
          f"{N_HEADS} Q heads, {N_KV} KV heads)")
    print(f"Prompt: {prompt}")
    print("-" * 60)

    t0 = time.time()
    output = generate(prompt, max_tokens=40, temperature=0.0)
    elapsed = time.time() - t0

    print(f"Output: {output}")
    print(f"Time:   {elapsed:.2f}s")
    print()

    # Second demo with sampling
    prompt2 = "In machine learning, gradient descent"
    print(f"Prompt: {prompt2}")
    print("-" * 60)

    t0 = time.time()
    output2 = generate(prompt2, max_tokens=40, temperature=0.7, top_p=0.9)
    elapsed2 = time.time() - t0

    print(f"Output: {output2}")
    print(f"Time:   {elapsed2:.2f}s")
