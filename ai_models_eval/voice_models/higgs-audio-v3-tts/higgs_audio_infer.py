#!/usr/bin/env python3
"""Higgs Audio v3 TTS — Core inference engine.

Production-ready text-to-speech for bosonai/higgs-audio-v3-tts-4b.
Supports batch synthesis, incremental streaming, LLM token pipeline,
bilingual text (EN/CN), voice cloning, and automatic long-text splitting.
"""

import os
import re
import gc
import io
import wave
import struct
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

BOC_ID = 1024
EOC_ID = 1025
SAMPLE_RATE = 24000
POST_DECODE_SILENCE_SEC = 0.03
DEFAULT_PREROLL_TOKEN = "<|prosody:pause|>"
TARGET_RMS = 0.2
MAX_GAIN = 5.0
INCREMENTAL_GAIN = 1.5

SPLIT_THRESHOLD_CHARS = 120
SPLIT_TARGET_SECONDS = 12.0
SPLIT_MAX_WORDS = 28

STREAM_CROSSFADE_SAMPLES = 240
STREAM_HOLDBACK_SAMPLES = 120

CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+:[^|]+?\|>")
LEADING_CONTROL_TOKEN_RE = re.compile(r"\s*(<\|[^|]+:[^|]+?\|>)")
TERMINAL_PUNCT_CHARS = ".!?。！？"
WEAK_PUNCT_CHARS = ",;，；、:："
CLOSING_QUOTE_CHARS = "\"'\"'）)]》」』"
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

VOICE_PRESETS = {
    "alloy":  {"temperature": 0.75, "top_k": 50, "top_p": 0.95, "seed": 1234},
    "echo":   {"temperature": 0.80, "top_k": 50, "top_p": 0.95, "seed": 1235},
    "fable":  {"temperature": 0.70, "top_k": 50, "top_p": 0.92, "seed": 1236},
    "onyx":   {"temperature": 0.85, "top_k": 80, "top_p": 0.95, "seed": 1237},
    "nova":   {"temperature": 0.80, "top_k": 50, "top_p": None, "seed": 1238},
    "shimmer":{"temperature": 0.90, "top_k": 80, "top_p": 0.97, "seed": 1239},
}


def _estimate_text_seconds(piece):
    normalized = " ".join(piece.strip().split())
    if not normalized:
        return 0.0
    non_space = len(normalized.replace(" ", ""))
    words = len(normalized.split())
    cjk = len(_CJK_CHAR_RE.findall(normalized))
    if cjk >= max(8, int(non_space * 0.20)):
        return (cjk / 4.0) + ((non_space - cjk) / 12.0)
    return max(words / 1.8, non_space / 10.0)


def _needs_split(text):
    non_space = len(text.replace(" ", ""))
    return non_space > SPLIT_THRESHOLD_CHARS or _estimate_text_seconds(text) > SPLIT_TARGET_SECONDS


def _split_text(text):
    if not _needs_split(text):
        return [text.strip()]
    try:
        voice_models_dir = Path(__file__).resolve().parent.parent
        import sys
        if str(voice_models_dir) not in sys.path:
            sys.path.insert(0, str(voice_models_dir))
        from tts_utils import split_text_for_reanchor
        segments = split_text_for_reanchor(text, max_words=SPLIT_MAX_WORDS,
                                            target_seconds=SPLIT_TARGET_SECONDS)
        if segments:
            return segments
    except ImportError:
        logger.debug("tts_utils not available, using inline splitting")
    clean = " ".join(text.strip().split())
    sentence_re = re.compile(r"[^.!?。！？]+(?:[.!?。！？]+)?")
    sentences = [m.group(0).strip() for m in sentence_re.finditer(clean) if m.group(0).strip()]
    segments = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip() if current else sent
        if (current and (_estimate_text_seconds(candidate) > SPLIT_TARGET_SECONDS
                         or len(candidate.replace(" ", "")) > SPLIT_THRESHOLD_CHARS)):
            segments.append(current)
            current = sent
        else:
            current = candidate
    if current:
        segments.append(current)
    return [s.strip() for s in segments if s.strip()] or [clean.strip()]


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = None
    seed: int = None


class _SamplerState:
    __slots__ = ["num_codebooks", "delay_count", "eoc_countdown", "generation_done"]
    def __init__(self, num_codebooks):
        self.num_codebooks = num_codebooks
        self.delay_count = 0
        self.eoc_countdown = None
        self.generation_done = False


def _sample(logits_NV, temperature, top_p, top_k):
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


def _reverse_delay_pattern(delayed_LN):
    L, Nc = delayed_LN.shape
    T = L - (Nc - 1)
    out = torch.empty((T, Nc), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(Nc):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def _apply_delay_pattern(codes_TN):
    T, N = codes_TN.shape
    out = torch.full((T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype)
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def _ensure_terminal_punctuation(text):
    stripped = text.strip()
    if not stripped:
        return stripped
    suffix, body = "", stripped
    while body and body[-1] in CLOSING_QUOTE_CHARS:
        suffix = body[-1] + suffix
        body = body[:-1].rstrip()
    if not body or body[-1] in TERMINAL_PUNCT_CHARS:
        return body + suffix
    if body[-1] in WEAK_PUNCT_CHARS:
        body = body[:-1].rstrip()
    cjk = len(_CJK_CHAR_RE.findall(body))
    non_space = len(body.replace(" ", ""))
    terminal = "。" if cjk >= max(4, int(non_space * 0.20)) else "."
    return body + terminal + suffix


def _add_generation_preroll(text, preroll_token=DEFAULT_PREROLL_TOKEN):
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
    if len(wav_np) == 0:
        return wav_np
    abs_wav = np.abs(wav_np)
    min_samples = int(min_silence_sec * SAMPLE_RATE)
    cut_point = len(wav_np)
    for i in range(len(wav_np) - 1, min_samples - 1, -1):
        if abs_wav[i] > threshold:
            cut_point = i + min_samples
            break
    return wav_np[:cut_point]


def _compute_max_steps(text_input):
    words = len(text_input.split())
    chars = len(text_input.replace(" ", ""))
    return min(1024, max(192, max(int(words * 4), int(chars * 6)) + 160))


def _normalize_audio(wav_np):
    rms = float(np.sqrt(np.mean(wav_np ** 2)))
    if rms > 0:
        gain = min(TARGET_RMS / rms, MAX_GAIN)
        wav_np = np.clip(wav_np * gain, -1.0, 1.0)
    return wav_np


class HiggsTTS:
    def __init__(self, model_path, device="cuda:0", dtype=torch.bfloat16):
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        self.device = device
        self.dtype = dtype
        self._device_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype, device_map=device,
        ).eval()
        self.model.requires_grad_(False)
        self.num_codebooks = self.model.num_codebooks
        logger.info("HiggsTTS loaded: device=%s, codebooks=%d, dtype=%s",
                     self.model.device, self.num_codebooks, dtype)

    def encode_reference_audio(self, reference_audio, reference_sr=None):
        if isinstance(reference_audio, str):
            with wave.open(reference_audio, "rb") as wf:
                ref_sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            ref_tensor = torch.from_numpy(
                np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            ).float()
        elif isinstance(reference_audio, np.ndarray):
            ref_tensor = torch.from_numpy(reference_audio).float()
            ref_sr = reference_sr or SAMPLE_RATE
        else:
            ref_tensor = reference_audio.float()
            ref_sr = reference_sr or SAMPLE_RATE
        with torch.inference_mode():
            codes_TN = self.model._encode_reference(ref_tensor, ref_sr)
        delayed_ref = _apply_delay_pattern(codes_TN.cpu())
        logger.info("Reference encoded: %d frames, %d codebooks",
                     codes_TN.shape[0], codes_TN.shape[1])
        self._clean_vram()
        return delayed_ref

    def synthesize(self, text_input, output_path=None, voice="alloy",
                   temperature=None, top_k=None, top_p=None, seed=None,
                   max_steps=None, reference_audio=None, reference_text=None,
                   add_preroll=True, close_utterance=True):
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        segments = _split_text(text_input)
        if len(segments) > 1:
            logger.info("Long text split into %d segments", len(segments))
        chunk_wavs = []
        for i, segment in enumerate(segments):
            if len(segments) > 1:
                logger.info("Synthesizing segment %d/%d: %s", i+1, len(segments),
                            segment[:60].replace('\n', ' '))
            steps = _compute_max_steps(segment) if max_steps is None else max_steps
            rows = self._run_ar_generation(segment, cfg, steps, reference_audio,
                                            reference_text, add_preroll, close_utterance)
            wav_np = self._decode_rows(rows)
            chunk_wavs.append(wav_np)
            self._clean_vram()
        combined = np.concatenate(chunk_wavs)
        combined = _normalize_audio(combined)
        if output_path is None:
            import time
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"/tmp/higgs_{ts}.wav"
        self._write_wav(combined, output_path)
        duration = len(combined) / SAMPLE_RATE
        logger.info("Saved %s (%.1fs, %d segment(s))", output_path, duration, len(segments))
        return output_path, duration

    def stream(self, text_input, voice="alloy", chunk_size=200,
               temperature=None, top_k=None, top_p=None, seed=None,
               max_steps=None, reference_audio=None, reference_text=None,
               add_preroll=True, close_utterance=True, decode_every=50):
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        segments = _split_text(text_input)
        if len(segments) > 1:
            logger.info("Streaming %d segments", len(segments))
        for i, segment in enumerate(segments):
            if len(segments) > 1:
                logger.info("Streaming segment %d/%d", i+1, len(segments))
            steps = _compute_max_steps(segment) if max_steps is None else max_steps
            yield from self._stream_incremental(segment, cfg, steps, chunk_size,
                                                 reference_audio, reference_text,
                                                 add_preroll, close_utterance,
                                                 decode_every)
            self._clean_vram()

    def stream_from_tokens(self, text_token_generator, voice="alloy",
                           chunk_size=200, temperature=None, top_k=None,
                           top_p=None, seed=None, reference_audio=None,
                           reference_text=None, max_buffer_sec=5.0, decode_every=50):
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        sentence_re = re.compile(r'([.!?。\u203f！\u203f]+[\s"]*)')
        buffer = ""
        def flush(text):
            if not text.strip():
                return
            text = text.strip()
            logger.info("Pipeline synthesizing: %s", text[:80])
            steps = _compute_max_steps(text)
            yield from self._stream_incremental(text, cfg, steps, chunk_size,
                                                 reference_audio, reference_text,
                                                 True, True, decode_every)
            self._clean_vram()
        try:
            for token in text_token_generator:
                buffer += token
                if _estimate_text_seconds(buffer) >= max_buffer_sec and buffer.strip():
                    yield from flush(buffer)
                    buffer = ""
                match = sentence_re.search(buffer)
                if match:
                    utterance = buffer[:match.end()]
                    buffer = buffer[match.end():]
                    yield from flush(utterance)
        finally:
            if buffer.strip():
                logger.info("Pipeline flushing: %s", buffer[:80])
                yield from flush(buffer)

    def _resolve_config(self, voice, temperature, top_k, top_p, seed):
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
        N = self.num_codebooks
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
        delayed_ref = None
        if reference_audio is not None:
            delayed_ref = (reference_audio if isinstance(reference_audio, torch.Tensor)
                          else self.encode_reference_audio(reference_audio))
        source = (_ensure_terminal_punctuation(text_input) if close_utterance
                  else text_input.strip())
        gen_text = (_add_generation_preroll(source) if add_preroll else source)
        num_ref = 0 if delayed_ref is None else delayed_ref.shape[0]
        prompt_ids = self.model._build_prompt_ids(
            self.tokenizer, gen_text, num_ref_tokens=num_ref, reference_text=reference_text)
        state = _SamplerState(N)
        rows = []
        with torch.inference_mode():
            embeds = self.model._prefill_embeds(prompt_ids, delayed_ref)
            out = self.model.model(inputs_embeds=embeds, use_cache=True)
            past, hidden, pos = out.past_key_values, out.last_hidden_state[:, -1, :], embeds.shape[1]
            for step in range(max_steps):
                logits = self.model.audio_head(hidden).to(torch.float32)[0]
                codes = _sampler_step(logits, state, cfg.temperature, cfg.top_p, cfg.top_k)
                if state.generation_done:
                    logger.debug("EOC at step %d", step)
                    break
                rows.append(codes.cpu())
                step_emb = self.model.audio_embedding(codes.unsqueeze(0)).unsqueeze(1)
                cpos = torch.tensor([pos], device=self.model.device)
                out = self.model.model(inputs_embeds=step_emb.to(embeds.dtype),
                                       past_key_values=past, use_cache=True, cache_position=cpos)
                past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
                del logits, codes, step_emb, cpos
                pos += 1
        if len(rows) >= max_steps - N:
            logger.warning("Hit max_steps=%d cap for: %s", max_steps, text_input[:80])
        if len(rows) < N:
            logger.warning("Too few rows (%d/%d) for: %s", len(rows), N, text_input[:80])
        logger.debug("Generated %d rows (max_steps=%d)", len(rows), max_steps)
        return rows

    def _stream_incremental(self, text_input, cfg, max_steps, chunk_size,
                             reference_audio, reference_text,
                             add_preroll, close_utterance, decode_every):
        """Incremental AR generation with sample-level crossfade for gapless output."""
        N = self.num_codebooks
        FADE = STREAM_CROSSFADE_SAMPLES
        HOLD = STREAM_HOLDBACK_SAMPLES

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        delayed_ref = None
        if reference_audio is not None:
            delayed_ref = (reference_audio if isinstance(reference_audio, torch.Tensor)
                          else self.encode_reference_audio(reference_audio))

        source = (_ensure_terminal_punctuation(text_input) if close_utterance
                  else text_input.strip())
        gen_text = (_add_generation_preroll(source) if add_preroll else source)

        num_ref = 0 if delayed_ref is None else delayed_ref.shape[0]
        prompt_ids = self.model._build_prompt_ids(
            self.tokenizer, gen_text, num_ref_tokens=num_ref, reference_text=reference_text)

        state = _SamplerState(N)
        rows = []
        samples_emitted = 0
        prev_tail = None

        def _yield_pcm(segment):
            if len(segment) == 0:
                return
            seg = np.clip(segment * INCREMENTAL_GAIN, -1.0, 1.0)
            pcm = np.clip(seg * 32767.0, -32768, 32767).astype(np.int16)
            off = 0
            while off < len(pcm):
                e = min(len(pcm), off + chunk_size // 2)
                yield struct.pack("<%dh" % (e - off), *pcm[off:e])
                off = e

        def _crossfade(old_tail, new_head, fade_len):
            fl = min(fade_len, len(old_tail), len(new_head))
            if fl <= 0:
                return np.concatenate([old_tail, new_head])
            fade_out = np.linspace(1.0, 0.0, fl)
            fade_in = np.linspace(0.0, 1.0, fl)
            blended = old_tail[-fl:] * fade_out + new_head[:fl] * fade_in
            return np.concatenate([old_tail[:-fl], blended, new_head[fl:]])

        def _emit_batch(is_final=False):
            nonlocal samples_emitted, prev_tail
            if len(rows) < N:
                return
            delayed_LN = torch.stack(rows, dim=0)
            codes_TN = _reverse_delay_pattern(delayed_LN)
            wav = self.model._decode_codes(codes_TN.to(self.model.device))
            wav_np = wav.numpy().astype(np.float32)
            total_samples = len(wav_np)
            if total_samples <= samples_emitted:
                del delayed_LN, codes_TN, wav, wav_np
                return
            emit_to = total_samples if is_final else max(samples_emitted + 1, total_samples - HOLD)
            new_audio = wav_np[samples_emitted:emit_to].copy()
            if prev_tail is not None and len(prev_tail) > 0:
                new_audio = _crossfade(prev_tail, new_audio, FADE)
            if not is_final and len(new_audio) > FADE:
                prev_tail = new_audio[-FADE:].copy()
                new_audio = new_audio[:-FADE]
            else:
                prev_tail = None
            if len(new_audio) > 0:
                yield from _yield_pcm(new_audio)
                samples_emitted = emit_to
            del delayed_LN, codes_TN, wav, wav_np

        with torch.inference_mode():
            embeds = self.model._prefill_embeds(prompt_ids, delayed_ref)
            out = self.model.model(inputs_embeds=embeds, use_cache=True)
            past, hidden, pos = out.past_key_values, out.last_hidden_state[:, -1, :], embeds.shape[1]

            for step in range(max_steps):
                logits = self.model.audio_head(hidden).to(torch.float32)[0]
                codes = _sampler_step(logits, state, cfg.temperature, cfg.top_p, cfg.top_k)
                if state.generation_done:
                    logger.debug("EOC at step %d", step)
                    break
                rows.append(codes.cpu())
                if len(rows) >= N + decode_every and (len(rows) - N) % decode_every == 0:
                    yield from _emit_batch(is_final=False)
                    logger.debug("Emitted %.1fs (step %d, rows %d)",
                                 samples_emitted / SAMPLE_RATE, step, len(rows))
                step_emb = self.model.audio_embedding(codes.unsqueeze(0)).unsqueeze(1)
                cpos = torch.tensor([pos], device=self.model.device)
                out = self.model.model(inputs_embeds=step_emb.to(embeds.dtype),
                                       past_key_values=past, use_cache=True, cache_position=cpos)
                past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
                del logits, codes, step_emb, cpos
                pos += 1

            if rows:
                yield from _emit_batch(is_final=True)

            logger.info("Stream done: %d rows, %.1fs total audio",
                        len(rows), samples_emitted / SAMPLE_RATE)

    def _decode_rows(self, rows):
        with torch.inference_mode():
            delayed_LN = torch.stack(rows, dim=0)
            codes_TN = _reverse_delay_pattern(delayed_LN)
            wav = self.model._decode_codes(codes_TN.to(self.model.device))
            wav_np = wav.numpy().astype(np.float32)
        silence = np.zeros(int(POST_DECODE_SILENCE_SEC * SAMPLE_RATE), dtype=wav_np.dtype)
        wav_np = np.concatenate([silence, wav_np])
        wav_np = _trim_trailing_silence(wav_np)
        wav_np = _normalize_audio(wav_np)
        return wav_np

    def _write_wav(self, wav_np, output_path):
        pcm = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

    def _clean_vram(self):
        if hasattr(self.model, "_audio_codec") and self.model._audio_codec is not None:
            self.model._audio_codec.cpu()
            self.model._audio_codec = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def get_vram_info(self):
        torch.cuda.synchronize(self._device_idx)
        free = torch.cuda.mem_get_info(self._device_idx)[0] / 1e9
        total = torch.cuda.get_device_properties(self._device_idx).total_memory / 1e9
        alloc = torch.cuda.memory_allocated(self._device_idx) / 1e6
        peak = torch.cuda.max_memory_allocated(self._device_idx) / 1e6
        logger.info("VRAM: %.1f/%.0f GB free, Alloc: %.0f MB, Peak: %.0f MB",
                     free, total, alloc, peak)
