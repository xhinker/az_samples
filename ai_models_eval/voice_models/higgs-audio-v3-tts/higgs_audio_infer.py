#!/usr/bin/env python3
"""Higgs Audio v3 TTS — Core inference engine.

Provides batch and streaming text-to-speech inference for the
bosonai/higgs-audio-v3-tts-4b model. Designed to run on a single
GPU (~10 GB VRAM) in bfloat16.

Usage:
    from higgs_audio_infer import HiggsTTS

    tts = HiggsTTS(model_path="/path/to/model", device="cuda:0")
    
    # Batch: full text -> WAV file
    wav_path, duration = tts.synthesize("Hello world.", output_path="/tmp/out.wav")
    
    # Streaming: full text -> PCM byte generator
    for pcm_chunk in tts.stream("Hello world."):
        audio_player.write(pcm_chunk)
    
    # Pipeline: token generator -> PCM byte stream (for LLM integration)
    for pcm_chunk in tts.stream_from_tokens(llm_token_generator):
        audio_player.write(pcm_chunk)
"""

import os
import re
import gc
import time
import wave
import struct
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Constants ---
BOC_ID = 1024
EOC_ID = 1025
SAMPLE_RATE = 24000
POST_DECODE_SILENCE_SEC = 0.03
DEFAULT_PREROLL_TOKEN = "<|prosody:pause|>"

CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+:[^|]+?\|>")
LEADING_CONTROL_TOKEN_RE = re.compile(r"\s*(<\|[^|]+:[^|]+?\|>)")
TERMINAL_PUNCT_CHARS = ".!?。！？"
WEAK_PUNCT_CHARS = ",;，；、:："
CLOSING_QUOTE_CHARS = "\"'\"'）)]》」』"
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# Voice presets (OpenAI-compatible names -> temperature/seed mapping)
VOICE_PRESETS = {
    "alloy":  {"temperature": 0.75, "top_k": 50, "top_p": 0.95, "seed": 1234},
    "echo":   {"temperature": 0.80, "top_k": 50, "top_p": 0.95, "seed": 1235},
    "fable":  {"temperature": 0.70, "top_k": 50, "top_p": 0.92, "seed": 1236},
    "onyx":   {"temperature": 0.85, "top_k": 80, "top_p": 0.95, "seed": 1237},
    "nova":   {"temperature": 0.80, "top_k": 50, "top_p": None, "seed": 1238},
    "shimmer":{"temperature": 0.90, "top_k": 80, "top_p": 0.97, "seed": 1239},
}


@dataclass
class GenerationConfig:
    """Sampling configuration for TTS generation."""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = None
    seed: int = None
    max_steps: int = None


class _SamplerState:
    """Multi-codebook delay sampler state machine."""
    __slots__ = ["num_codebooks", "delay_count", "eoc_countdown", "generation_done"]

    def __init__(self, num_codebooks):
        self.num_codebooks = num_codebooks
        self.delay_count = 0
        self.eoc_countdown = None
        self.generation_done = False


def _sample(logits_NV, temperature, top_p, top_k):
    """Official sampling: temperature scaling -> top-k -> top-p -> multinomial."""
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


# --- Helper functions (module-level) ---

def _reverse_delay_pattern(delayed_LN):
    """Undo the delay pattern: [L, N] -> [T, N] where T = L - N + 1."""
    L, Nc = delayed_LN.shape
    T = L - (Nc - 1)
    out = torch.empty((T, Nc), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(Nc):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def _apply_delay_pattern(codes_TN):
    """Apply delay pattern: [T, N] -> [T + N - 1, N], BOC/EOC padded."""
    T, N = codes_TN.shape
    out = torch.full((T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype)
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def _is_cjk_heavy(piece):
    non_space_chars = len(piece.replace(" ", ""))
    cjk_chars = len(_CJK_CHAR_RE.findall(piece))
    return cjk_chars >= max(8, int(non_space_chars * 0.20))


def _ensure_terminal_punctuation(text):
    """Close an utterance so Higgs emits EOC."""
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
    terminal = "。" if _is_cjk_heavy(body) else "."
    return body + terminal + suffix


def _add_generation_preroll(text, preroll_token=DEFAULT_PREROLL_TOKEN):
    """Insert a prosody pause before the first spoken token."""
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


def _trim_trailing_silence(wav_np, threshold=0.01, min_silence_sec=0.5):
    """Trim trailing silence/garbage from end of decoded audio."""
    if len(wav_np) == 0:
        return wav_np
    abs_wav = np.abs(wav_np)
    min_samples = int(min_silence_sec * SAMPLE_RATE)
    cut_point = len(wav_np)
    for i in range(len(wav_np) - 1, min_samples - 1, -1):
        if abs_wav[i] > threshold:
            cut_point = i + min_samples
            break
    if cut_point < len(wav_np):
        trimmed_sec = (len(wav_np) - cut_point) / SAMPLE_RATE
    return wav_np[:cut_point]


def _compute_max_steps(text_input):
    """Auto-calculate max AR steps from text length."""
    words = len(text_input.split())
    chars = len(text_input.replace(" ", ""))
    return min(1024, max(192, max(int(words * 4), int(chars * 6)) + 160))


# --- Main class ---

class HiggsTTS:
    """Higgs Audio v3 TTS inference engine.

    Loads the model once, then supports batch synthesis, streaming
    synthesis, and real-time token-to-audio pipeline.

    Args:
        model_path: Path to higgs-audio-v3-tts-4b model directory.
        device: GPU device string, e.g. "cuda:0".
        dtype: Model precision, torch.bfloat16 (default) or torch.float16.
    """

    def __init__(self, model_path, device="cuda:0", dtype=torch.bfloat16):
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        self.device = device
        self.dtype = dtype
        self.num_codebooks = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device,
        ).eval()
        self.model.requires_grad_(False)

        self.num_codebooks = self.model.num_codebooks
        self._device_idx = int(str(self.model.device).split(":")[-1])

        print(f"[HiggsTTS] Model loaded on {self.model.device}, "
              f"{self.num_codebooks} codebooks, dtype={dtype}")

    # ---- Public API ----

    def encode_reference_audio(self, reference_audio, reference_sr=None):
        """Encode a reference WAV for voice cloning (encode once, reuse).

        Args:
            reference_audio: Path to WAV file, numpy array, or torch tensor.
            reference_sr: Sample rate (auto-detected from WAV path).

        Returns:
            delayed_ref tensor ready to pass as reference_audio to synthesize/stream.
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
            ref_sr = reference_sr or SAMPLE_RATE
        else:
            ref_tensor = reference_audio.float()
            ref_sr = reference_sr or SAMPLE_RATE

        with torch.inference_mode():
            codes_TN = self.model._encode_reference(ref_tensor, ref_sr)
        delayed_ref = _apply_delay_pattern(codes_TN.cpu())
        print(f"  [HiggsTTS] Reference encoded: {codes_TN.shape[0]} frames, "
              f"{codes_TN.shape[1]} codebooks (delayed: {delayed_ref.shape[0]} rows)")
        self._clean_vram()
        return delayed_ref

    def synthesize(self, text_input, output_path=None, voice="alloy",
                   temperature=None, top_k=None, top_p=None, seed=None,
                   max_steps=None, reference_audio=None, reference_text=None,
                   add_preroll=True, close_utterance=True):
        """Batch synthesis: text -> WAV file.

        Args:
            text_input: Text with optional control tags.
            output_path: Output WAV path. None = auto-generated unique path.
            voice: Voice preset name (alloy/echo/fable/onyx/nova/shimmer).
            temperature, top_k, top_p, seed: Override voice preset defaults.
            max_steps: AR step cap. None = auto from text length.
            reference_audio: Pre-encoded delayed_ref tensor for voice cloning.
            reference_text: Transcript of reference audio.
            add_preroll: Add prosody pause before first token.
            close_utterance: Ensure terminal punctuation.

        Returns:
            (wav_path, duration_seconds) tuple.
        """
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        if max_steps is None:
            max_steps = _compute_max_steps(text_input)

        if output_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"/tmp/higgs_{ts}_{id(text_input):08x}.wav"

        rows = self._run_ar_generation(text_input, cfg, max_steps,
                                        reference_audio, reference_text,
                                        add_preroll, close_utterance)

        wav_np = self._decode_rows(rows)
        self._write_wav(wav_np, output_path)
        duration_s = len(wav_np) / SAMPLE_RATE
        print(f"  [HiggsTTS] Saved {output_path} ({duration_s:.1f}s audio)")
        self._clean_vram()
        return output_path, duration_s

    def stream(self, text_input, voice="alloy", chunk_size=200,
               temperature=None, top_k=None, top_p=None, seed=None,
               max_steps=None, reference_audio=None, reference_text=None,
               add_preroll=True, close_utterance=True, decode_every=50):
        """Incremental streaming: yields PCM as AR loop generates rows.

        Every `decode_every` AR rows, accumulated codes are decoded through
        the vocoder and yielded as PCM bytes. This means audio starts flowing
        to the caller while the model is still generating the rest.

        Args:
            Same as synthesize(), plus:
            chunk_size: PCM bytes per yield (default 200 = ~4ms audio).
            decode_every: Yield audio every N rows (default 50 = ~2s).

        Yields:
            bytes: PCM16LE audio chunks (24kHz, mono).
        """
        yield from self.stream_incremental(
            text_input=text_input, voice=voice, chunk_size=chunk_size,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
            max_steps=max_steps, reference_audio=reference_audio,
            reference_text=reference_text, add_preroll=add_preroll,
            close_utterance=close_utterance, decode_every=decode_every,
        )

    def stream_incremental(self, text_input, voice="alloy", chunk_size=960,
                               temperature=None, top_k=None, top_p=None, seed=None,
                               max_steps=None, reference_audio=None, reference_text=None,
                               add_preroll=True, close_utterance=True,
                               decode_every=50):
        """True incremental streaming: yields PCM as AR loop generates rows.

        Unlike stream() which generates ALL rows then decodes, this method
        yields audio chunks progressively during generation. The AR loop
        runs row-by-row, and every `decode_every` rows, the accumulated
        codes are decoded through the vocoder and yielded as PCM bytes.

        This enables real-time streaming where the user hears the first
        words while the model is still generating the rest.

        Args:
            text_input: Text to synthesize.
            voice: Voice preset name.
            chunk_size: PCM bytes per yield.
            temperature, top_k, top_p, seed: Sampling config.
            max_steps: AR step cap. None = auto.
            reference_audio: Pre-encoded delayed_ref tensor.
            reference_text: Transcript of reference audio.
            add_preroll: Add prosody pause before first token.
            close_utterance: Ensure terminal punctuation.
            decode_every: Decode and yield every N rows (default 50 = ~2s audio).
                          Lower = less latency but more vocoder overhead.

        Yields:
            bytes: PCM16LE audio chunks (24kHz, mono).
        """
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        if max_steps is None:
            max_steps = _compute_max_steps(text_input)

        N = self.num_codebooks

        # Set seed
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

        # Reference audio
        delayed_ref = None
        if reference_audio is not None:
            if isinstance(reference_audio, torch.Tensor):
                delayed_ref = reference_audio
            else:
                delayed_ref = self.encode_reference_audio(reference_audio)

        # Build prompt
        source_text = (_ensure_terminal_punctuation(text_input)
                       if close_utterance else text_input.strip())
        generation_text = (_add_generation_preroll(source_text)
                           if add_preroll else source_text)

        num_ref_tokens = 0 if delayed_ref is None else delayed_ref.shape[0]
        prompt_ids_list = self.model._build_prompt_ids(
            self.tokenizer, generation_text,
            num_ref_tokens=num_ref_tokens, reference_text=reference_text
        )

        state = _SamplerState(N)
        rows = []  # Accumulated codebook rows [L, N]
        samples_yielded = 0  # Total PCM samples already sent to client

        with torch.inference_mode():
            inputs_embeds = self.model._prefill_embeds(prompt_ids_list, delayed_ref)
            out = self.model.model(inputs_embeds=inputs_embeds, use_cache=True)
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]
            position = inputs_embeds.shape[1]

            for step in range(max_steps):
                logits_NV = self.model.audio_head(hidden_last).to(torch.float32)[0]
                codes_N = _sampler_step(logits_NV, state, cfg.temperature,
                                         cfg.top_p, cfg.top_k)

                if state.generation_done:
                    print(f"  [HiggsTTS] EOC at step {step}")
                    break

                rows.append(codes_N.cpu())

                # --- Incremental decode: every decode_every rows ---
                if len(rows) >= N + decode_every and (len(rows) - N) % decode_every == 0:
                    delayed_LN = torch.stack(rows, dim=0)  # [L, N]
                    codes_TN = _reverse_delay_pattern(delayed_LN)  # [T, N]
                    wav = self.model._decode_codes(codes_TN.to(self.model.device))
                    wav_np = wav.numpy().astype(np.float32)

                    # Only yield NEW samples (after last yield point)
                    new_start = samples_yielded
                    new_end = len(wav_np)
                    if new_end > new_start:
                        segment = wav_np[new_start:new_end]

                        # Normalize volume
                        rms = float(np.sqrt(np.mean(segment ** 2)))
                        if rms > 0:
                            gain = min(0.2 / rms, 5.0)
                            segment = np.clip(segment * gain, -1.0, 1.0)

                        pcm = np.clip(segment * 32767.0, -32768, 32767).astype(np.int16)
                        offset = 0
                        total = len(pcm)
                        while offset < total:
                            end = min(total, offset + chunk_size // 2)
                            yield struct.pack("<%dh" % (end - offset), *pcm[offset:end])
                            offset = end

                        samples_yielded = new_end
                        print(f"  [HiggsTTS] Stream: {new_end/SAMPLE_RATE:.1f}s audio yielded "
                              f"(step {step}, rows {len(rows)})")

                    # Free intermediates
                    del delayed_LN, codes_TN, wav, wav_np, segment, pcm

                # AR next step
                step_embed = self.model.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
                cache_pos = torch.tensor([position], device=self.model.device)
                out = self.model.model(
                    inputs_embeds=step_embed.to(inputs_embeds.dtype),
                    past_key_values=past, use_cache=True, cache_position=cache_pos,
                )
                past = out.past_key_values
                hidden_last = out.last_hidden_state[:, -1, :]

                del logits_NV, codes_N, step_embed, cache_pos
                position += 1

        # --- Final decode: yield remaining audio ---
        if len(rows) >= N:
            delayed_LN = torch.stack(rows, dim=0)
            codes_TN = _reverse_delay_pattern(delayed_LN)
            wav = self.model._decode_codes(codes_TN.to(self.model.device))
            wav_np = wav.numpy().astype(np.float32)

            if len(wav_np) > samples_yielded:
                segment = wav_np[samples_yielded:]

                # Prepend guard silence + trim trailing
                silence_samples = int(POST_DECODE_SILENCE_SEC * SAMPLE_RATE)
                silence = np.zeros(silence_samples, dtype=segment.dtype)
                segment = np.concatenate([silence, segment])
                segment = _trim_trailing_silence(segment)

                rms = float(np.sqrt(np.mean(segment ** 2)))
                if rms > 0:
                    gain = min(0.2 / rms, 5.0)
                    segment = np.clip(segment * gain, -1.0, 1.0)

                pcm = np.clip(segment * 32767.0, -32768, 32767).astype(np.int16)
                offset = 0
                total = len(pcm)
                while offset < total:
                    end = min(total, offset + chunk_size // 2)
                    yield struct.pack("<%dh" % (end - offset), *pcm[offset:end])
                    offset = end

        total_rows = len(rows)
        print(f"  [HiggsTTS] Done: {total_rows} rows generated")
        self._clean_vram()

    def stream_from_tokens(self, text_token_generator, voice="alloy",
                           chunk_size=200, temperature=None, top_k=None,
                           top_p=None, seed=None, reference_audio=None,
                           reference_text=None, max_buffer_sec=5.0):
        """Real-time pipeline: LLM text tokens -> PCM audio stream.

        Buffers incoming tokens, splits at sentence boundaries,
        synthesizes each utterance, yields PCM bytes.

        Args:
            text_token_generator: Generator yielding text tokens (str).
            voice, temperature, top_k, top_p, seed: Sampling config.
            chunk_size: PCM bytes per yield.
            reference_audio: Pre-encoded delayed_ref tensor.
            reference_text: Transcript of reference audio.
            max_buffer_sec: Max buffer time before force flush.

        Yields:
            bytes: PCM16LE audio chunks (24kHz, mono).
        """
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        _SENTENCE_END_RE = re.compile(r'([.!?。\u203f！\u203f]+[\s"]*)')
        buffer = ""

        def estimate_secs(text):
            chars = len(text.replace(" ", ""))
            cjk = len(_CJK_CHAR_RE.findall(text))
            if cjk >= max(8, int(chars * 0.20)):
                return cjk / 4.0 + (chars - cjk) / 12.0
            return max(len(text.split()) / 1.8, chars / 10.0)

        def flush(text):
            if not text.strip():
                return
            text = text.strip()
            print(f"  [HiggsTTS] Synthesizing: {text[:80]}{'...' if len(text) > 80 else ''}")
            for pcm_chunk in self.stream(
                text_input=text,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                seed=cfg.seed,
                reference_audio=reference_audio,
                reference_text=reference_text,
                chunk_size=chunk_size,
            ):
                yield pcm_chunk

        try:
            for token in text_token_generator:
                buffer += token
                buffer_time = estimate_secs(buffer)

                match = _SENTENCE_END_RE.search(buffer)
                if match:
                    utterance = buffer[:match.end()]
                    buffer = buffer[match.end():]
                    buffer_time = estimate_secs(buffer)
                    yield from flush(utterance)

                if buffer_time >= max_buffer_sec and buffer.strip():
                    yield from flush(buffer)
                    buffer = ""
                    buffer_time = 0.0
        finally:
            if buffer.strip():
                print(f"  [HiggsTTS] Flushing: {buffer[:80]}")
                yield from flush(buffer)

    # ---- Internal methods ----

    def _resolve_config(self, voice, temperature, top_k, top_p, seed):
        """Merge voice preset with explicit overrides."""
        preset = VOICE_PRESETS.get(voice, VOICE_PRESETS["alloy"])
        return GenerationConfig(
            temperature=temperature if temperature is not None else preset["temperature"],
            top_k=top_k if top_k is not None else preset["top_k"],
            top_p=top_p if top_p is not None else preset["top_p"],
            seed=seed if seed is not None else preset["seed"],
        )

    def _run_ar_generation(self, text_input, cfg, max_steps,
                            reference_audio, reference_text,
                            add_preroll, close_utterance):
        """Run the full AR generation loop. Returns list of codebook rows."""
        N = self.num_codebooks

        # Set seed
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

        # Reference audio
        delayed_ref = None
        if reference_audio is not None:
            if isinstance(reference_audio, torch.Tensor):
                delayed_ref = reference_audio
            else:
                delayed_ref = self.encode_reference_audio(reference_audio)

        # Build prompt
        source_text = (_ensure_terminal_punctuation(text_input)
                       if close_utterance else text_input.strip())
        generation_text = (_add_generation_preroll(source_text)
                           if add_preroll else source_text)

        num_ref_tokens = 0 if delayed_ref is None else delayed_ref.shape[0]
        prompt_ids_list = self.model._build_prompt_ids(
            self.tokenizer, generation_text,
            num_ref_tokens=num_ref_tokens, reference_text=reference_text
        )

        state = _SamplerState(N)
        rows = []

        with torch.inference_mode():
            inputs_embeds = self.model._prefill_embeds(prompt_ids_list, delayed_ref)
            out = self.model.model(inputs_embeds=inputs_embeds, use_cache=True)
            past = out.past_key_values
            hidden_last = out.last_hidden_state[:, -1, :]
            position = inputs_embeds.shape[1]

            for step in range(max_steps):
                logits_NV = self.model.audio_head(hidden_last).to(torch.float32)[0]
                codes_N = _sampler_step(logits_NV, state, cfg.temperature,
                                         cfg.top_p, cfg.top_k)

                if state.generation_done:
                    print(f"  [HiggsTTS] EOC at step {step}")
                    break

                rows.append(codes_N.cpu())

                step_embed = self.model.audio_embedding(codes_N.unsqueeze(0)).unsqueeze(1)
                cache_pos = torch.tensor([position], device=self.model.device)
                out = self.model.model(
                    inputs_embeds=step_embed.to(inputs_embeds.dtype),
                    past_key_values=past, use_cache=True, cache_position=cache_pos,
                )
                past = out.past_key_values
                hidden_last = out.last_hidden_state[:, -1, :]

                del logits_NV, codes_N, step_embed, cache_pos
                position += 1

        hit_cap = len(rows) >= max_steps - N
        if hit_cap:
            print(f"  [HiggsTTS] WARNING: hit max_steps={max_steps} cap!")
        if len(rows) < N:
            print(f"  [HiggsTTS] WARNING: too few rows ({len(rows)}/{N})")

        print(f"  [HiggsTTS] Generated {len(rows)} rows (max_steps={max_steps})")
        return rows

    def _decode_rows(self, rows):
        """Decode codebook rows -> processed numpy audio array."""
        with torch.inference_mode():
            delayed_LN = torch.stack(rows, dim=0)
            codes_TN = _reverse_delay_pattern(delayed_LN)
            wav = self.model._decode_codes(codes_TN.to(self.model.device))
            wav_np = wav.numpy().astype(np.float32)

        # Prepend guard silence
        silence_samples = int(POST_DECODE_SILENCE_SEC * SAMPLE_RATE)
        silence = np.zeros(silence_samples, dtype=wav_np.dtype)
        wav_np = np.concatenate([silence, wav_np])

        # Trim trailing silence
        wav_np = _trim_trailing_silence(wav_np)

        # Normalize volume to ~-14 dBFS
        current_rms = float(np.sqrt(np.mean(wav_np ** 2)))
        if current_rms > 0:
            gain = min(0.2 / current_rms, 5.0)
            wav_np = np.clip(wav_np * gain, -1.0, 1.0)

        return wav_np

    def _write_wav(self, wav_np, output_path):
        """Write numpy audio array to WAV file."""
        pcm = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

    def _clean_vram(self):
        """Aggressively reclaim VRAM."""
        if hasattr(self.model, "_audio_codec") and self.model._audio_codec is not None:
            self.model._audio_codec.cpu()
            self.model._audio_codec = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def get_vram_info(self):
        """Print current VRAM usage."""
        torch.cuda.synchronize(self._device_idx)
        free_gb = torch.cuda.mem_get_info(self._device_idx)[0] / 1e9
        total_gb = torch.cuda.get_device_properties(self._device_idx).total_memory / 1e9
        alloc_mb = torch.cuda.memory_allocated(self._device_idx) / 1e6
        reserved_mb = torch.cuda.memory_reserved(self._device_idx) / 1e6
        peak_mb = torch.cuda.max_memory_allocated(self._device_idx) / 1e6
        print(f"  [HiggsTTS] VRAM: {free_gb:.1f}/{total_gb:.0f} GB free | "
              f"Alloc: {alloc_mb:.0f} MB | Reserved: {reserved_mb:.0f} MB | "
              f"Peak: {peak_mb:.0f} MB")
