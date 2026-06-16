#!/usr/bin/env python3
"""Higgs Audio v3 TTS — Core inference engine.

Production-ready text-to-speech for bosonai/higgs-audio-v3-tts-4b.
Supports batch synthesis, incremental streaming, LLM token pipeline,
bilingual text (EN/CN), voice cloning, and automatic long-text splitting.

Usage:
    from higgs_audio_infer import HiggsTTS

    tts = HiggsTTS(model_path="/path/to/model", device="cuda:0")

    # Batch: auto-splits long text, returns single WAV
    path, dur = tts.synthesize("A very long story in English or 中文...")

    # Streaming: yields PCM bytes as AR loop generates (incremental)
    for pcm in tts.stream("Hello world."):
        player.write(pcm)

    # LLM pipeline: token generator -> PCM stream
    for pcm in tts.stream_from_tokens(llm_generator):
        player.write(pcm)
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

# =============================================================================
# Constants
# =============================================================================

# Multi-codebook delay sampler special tokens
BOC_ID = 1024   # Begin-Of-Codebook (padding token for delay pattern)
EOC_ID = 1025   # End-Of-Codebook (generation stop signal)

# Audio format
SAMPLE_RATE = 24000
POST_DECODE_SILENCE_SEC = 0.03  # guard silence prepended before vocoder output

# Generation defaults
DEFAULT_PREROLL_TOKEN = "<|prosody:pause|>"  # inserted before first spoken token

# Volume normalization
TARGET_RMS = 0.2          # ~-14 dBFS for final batch output
MAX_GAIN = 5.0            # cap gain to avoid noise amplification
INCREMENTAL_GAIN = 1.5    # fixed gain for streaming (consistent across batches)

# Long-text splitting thresholds
SPLIT_THRESHOLD_CHARS = 120   # split if non-space chars exceed this
SPLIT_TARGET_SECONDS = 12.0   # target audio duration per segment
SPLIT_MAX_WORDS = 28          # hard cap on words per segment

# Streaming: sample-level overlap + crossfade for gapless playback
STREAM_CROSSFADE_SAMPLES = 240   # 10ms linear crossfade at each batch boundary
STREAM_HOLDBACK_SAMPLES  = 120   # holdback at right edge (vocoder unstable zone)

# Text processing regex
CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+:[^|]+?\|>")
LEADING_CONTROL_TOKEN_RE = re.compile(r"\s*(<\|[^|]+:[^|]+?\|>)")
TERMINAL_PUNCT_CHARS = ".!?。！？"
WEAK_PUNCT_CHARS = ",;，；、:："
CLOSING_QUOTE_CHARS = "\"'\"'）)]》」』"
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _normalize_cjk_spaces(text):
    """Remove spaces between adjacent CJK characters.

    Chinese text often has spurious spaces from OCR, copy-paste, or
    tokenization artifacts (e.g. '特 别' should be '特别'). These spaces
    are tokenized as pause tokens by the model, causing audible gaps.

    Preserves spaces between Latin characters and between Latin and CJK:
    - '特 别'       -> '特别'       (CJK-CJK space removed)
    - 'Hello 世界'  -> 'Hello 世界'  (Latin-CJK space kept)
    - 'Hello World' -> 'Hello World' (Latin-Latin space kept)
    - '你 好 世 界' -> '你好世界'   (multiple CJK-CJK spaces removed)
    """
    if not text or ' ' not in text:
        return text
    result = []
    for i, ch in enumerate(text):
        if ch == ' ' and i > 0 and i < len(text) - 1:
            prev_is_cjk = bool(_CJK_CHAR_RE.match(text[i - 1]))
            next_is_cjk = bool(_CJK_CHAR_RE.match(text[i + 1]))
            if prev_is_cjk and next_is_cjk:
                continue  # skip this space
        result.append(ch)
    return ''.join(result)


# OpenAI-compatible voice presets (name -> sampling params)
VOICE_PRESETS = {
    "alloy":  {"temperature": 0.75, "top_k": 50, "top_p": 0.95, "seed": 1234},
    "echo":   {"temperature": 0.80, "top_k": 50, "top_p": 0.95, "seed": 1235},
    "fable":  {"temperature": 0.70, "top_k": 50, "top_p": 0.92, "seed": 1236},
    "onyx":   {"temperature": 0.85, "top_k": 80, "top_p": 0.95, "seed": 1237},
    "nova":   {"temperature": 0.80, "top_k": 50, "top_p": None, "seed": 1238},
    "shimmer":{"temperature": 0.90, "top_k": 80, "top_p": 0.97, "seed": 1239},
}


# =============================================================================
# Text splitting — handles long text for EN/CN bilingual
# =============================================================================

def _estimate_text_seconds(piece):
    """Estimate spoken duration for a text piece (EN/CN aware).

    Chinese characters speak at ~4 chars/sec, English at ~1.8 words/sec.
    For mixed text, weights by CJK character ratio.
    """
    normalized = " ".join(piece.strip().split())
    if not normalized:
        return 0.0
    non_space = len(normalized.replace(" ", ""))
    words = len(normalized.split())
    cjk = len(_CJK_CHAR_RE.findall(normalized))
    if cjk >= max(8, int(non_space * 0.20)):
        # CJK-heavy: ~4 chars/sec for CJK, ~12 chars/sec for non-CJK
        return (cjk / 4.0) + ((non_space - cjk) / 12.0)
    # English-heavy: ~1.8 words/sec or ~10 non-space chars/sec
    return max(words / 1.8, non_space / 10.0)


def _needs_split(text):
    """Check if text is long enough to warrant splitting into segments."""
    non_space = len(text.replace(" ", ""))
    return non_space > SPLIT_THRESHOLD_CHARS or _estimate_text_seconds(text) > SPLIT_TARGET_SECONDS


def _split_text(text):
    """Split long text into TTS-friendly segments (EN/CN bilingual).

    Tries tts_utils.split_text_for_reanchor first (from the original Higgs
    codebase). Falls back to inline sentence-based splitting if unavailable.

    Each segment is sized to fit comfortably within max_steps=1024 AR limit.
    """
    if not _needs_split(text):
        return [text.strip()]

    # Prefer tts_utils from the original Higgs codebase
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

    # Inline fallback: split at sentence boundaries, merge to target duration
    clean = " ".join(text.strip().split())
    sentence_re = re.compile(r"[^.!?。！？]+(?:[.!?。！？]+)?")
    sentences = [m.group(0).strip() for m in sentence_re.finditer(clean)
                 if m.group(0).strip()]

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


# =============================================================================
# AR sampler — multi-codebook delay pattern
# =============================================================================

@dataclass
class GenerationConfig:
    """Sampling parameters for one TTS generation."""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = None
    seed: int = None


class _StreamResult:
    """Holds PCM generator + tail for inter-segment crossfade.

    Used internally by stream() to pass tail samples between segments
    without polluting the PCM byte stream.
    """
    def __init__(self):
        self.tail = None
        self._chunks = []

    def add(self, pcm_bytes):
        self._chunks.append(pcm_bytes)

    def finalize(self, tail):
        self.tail = tail

    def chunks(self):
        return list(self._chunks)


class _SamplerState:
    """State machine for the multi-codebook delay sampler.

    The Higgs model uses N codebooks (typically 16). During generation,
    codebook c is delayed by c steps. This state machine handles:
    - The initial delay fill (first N steps produce BOC padding)
    - EOC detection (when codebook 0 emits EOC_ID)
    - Countdown after EOC (need N-2 more steps for remaining codebooks)
    """
    __slots__ = ["num_codebooks", "delay_count", "eoc_countdown", "generation_done"]

    def __init__(self, num_codebooks):
        self.num_codebooks = num_codebooks
        self.delay_count = 0          # how many delay steps remaining
        self.eoc_countdown = None     # steps remaining after EOC detected
        self.generation_done = False


def _sample(logits_NV, temperature, top_p, top_k):
    """Official sampling: temperature scaling -> top-k -> top-p -> multinomial.

    Args:
        logits_NV: [N_codebooks, vocab_size] logits from the model
        temperature: scaling factor (lower = more deterministic)
        top_p: nucleus sampling threshold
        top_k: top-k sampling limit

    Returns:
        [N_codebooks] sampled token IDs
    """
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
    """One AR step of the multi-codebook delay sampler. Mutates state.

    During the initial delay phase (first N steps), newly sampled codes
    are shifted down and BOC tokens fill the top. After delay, codes
    flow through normally until EOC is detected.
    """
    N = state.num_codebooks
    codes_N = _sample(logits_NV, temperature, top_p, top_k).to(torch.long)

    if state.delay_count < N:
        # Delay fill phase: shift codes down, pad top with BOC
        next_cb = state.delay_count + 1
        if next_cb < N:
            codes_N[next_cb:] = BOC_ID
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        # Post-EOC countdown: wait for remaining codebooks to flush
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_N[0].item()) == EOC_ID:
        # EOC detected in primary codebook
        if N <= 2:
            state.generation_done = True
        else:
            state.eoc_countdown = N - 2  # wait for remaining codebooks
    return codes_N


# =============================================================================
# Delay pattern — converts between delayed (AR) and raw (vocoder) formats
# =============================================================================

def _reverse_delay_pattern(delayed_LN):
    """Undo the delay pattern: [L, N] -> [T, N] where T = L - N + 1.

    The AR loop produces delayed rows where codebook c is offset by c steps.
    This function aligns all codebooks to the same time axis for vocoder input.
    """
    L, Nc = delayed_LN.shape
    T = L - (Nc - 1)
    out = torch.empty((T, Nc), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(Nc):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def _apply_delay_pattern(codes_TN):
    """Apply delay pattern: [T, N] -> [T + N - 1, N], BOC/EOC padded.

    Used when encoding reference audio for voice cloning. The raw codec
    frames must be delayed to match the AR generation format.
    """
    T, N = codes_TN.shape
    out = torch.full((T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype)
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID       # pad beginning with BOC
        out[c : c + T, c] = codes_TN[:, c]
    return out


# =============================================================================
# Text preprocessing — ensures proper generation boundaries
# =============================================================================

def _ensure_terminal_punctuation(text):
    """Close an utterance so Higgs emits EOC (End-Of-Codebook).

    Without terminal punctuation, the model may not generate EOC, causing
    the AR loop to run until max_steps (truncated audio). This function
    adds a period (English) or 。 (Chinese) if the text lacks one.
    """
    # First remove spurious spaces between CJK characters
    text = _normalize_cjk_spaces(text)
    stripped = text.strip()
    if not stripped:
        return stripped
    suffix, body = "", stripped
    # Strip trailing closing quotes to check the actual last character
    while body and body[-1] in CLOSING_QUOTE_CHARS:
        suffix = body[-1] + suffix
        body = body[:-1].rstrip()
    if not body or body[-1] in TERMINAL_PUNCT_CHARS:
        return body + suffix
    if body[-1] in WEAK_PUNCT_CHARS:
        body = body[:-1].rstrip()
    # Choose terminal punctuation based on language
    cjk = len(_CJK_CHAR_RE.findall(body))
    non_space = len(body.replace(" ", ""))
    terminal = "。" if cjk >= max(4, int(non_space * 0.20)) else "."
    return body + terminal + suffix


def _add_generation_preroll(text, preroll_token=DEFAULT_PREROLL_TOKEN):
    """Insert a prosody pause before the first spoken token.

    This gives the model a natural breathing room at the start,
    preventing abrupt onset. Skips existing control tokens.
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


def _trim_trailing_silence(wav_np, threshold=0.01, min_silence_sec=0.5):
    """Trim trailing silence/garbage from end of decoded audio.

    The vocoder may produce silence or artifacts after the last meaningful
    audio. This scans from the end and cuts at the last significant signal.
    """
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
    """Auto-calculate AR step cap from text length.

    Prevents runaway generation while allowing enough steps for the text.
    Capped at 1024 to bound VRAM usage.
    """
    words = len(text_input.split())
    chars = len(text_input.replace(" ", ""))
    return min(1024, max(192, max(int(words * 4), int(chars * 6)) + 160))


def _normalize_audio(wav_np):
    """Normalize audio to target RMS (~-14 dBFS) with gain cap."""
    rms = float(np.sqrt(np.mean(wav_np ** 2)))
    if rms > 0:
        gain = min(TARGET_RMS / rms, MAX_GAIN)
        wav_np = np.clip(wav_np * gain, -1.0, 1.0)
    return wav_np


# =============================================================================
# Main TTS engine
# =============================================================================

class HiggsTTS:
    """Higgs Audio v3 TTS inference engine.

    Loads the model once, then supports:
    - Batch synthesis (full text -> WAV file, auto-splits long text)
    - Incremental streaming (yields PCM as AR loop generates)
    - LLM token pipeline (real-time token -> audio for chatbots)
    - Voice cloning (reference audio encoding)

    Args:
        model_path: Path to higgs-audio-v3-tts-4b model directory.
        device: GPU device string, e.g. "cuda:0".
        dtype: Model precision, torch.bfloat16 (default) or torch.float16.
    """

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

    # ------------------------------------------------------------------
    # Voice cloning
    # ------------------------------------------------------------------

    def encode_reference_audio(self, reference_audio, reference_sr=None):
        """Encode a reference WAV for voice cloning (encode once, reuse).

        The reference audio is encoded into codec frames, then the delay
        pattern is applied so it matches the AR generation format.

        Args:
            reference_audio: Path to WAV file, numpy array, or torch tensor.
            reference_sr: Sample rate (auto-detected from WAV path).

        Returns:
            delayed_ref tensor ready to pass as reference_audio to synthesize/stream.
        """
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

    # ------------------------------------------------------------------
    # Public API: batch synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text_input, output_path=None, voice="alloy",
                   temperature=None, top_k=None, top_p=None, seed=None,
                   max_steps=None, reference_audio=None, reference_text=None,
                   add_preroll=True, close_utterance=True):
        """Batch synthesis: text -> WAV file. Auto-splits long text.

        For long text, splits into ~12s segments, generates each independently,
        concatenates the audio, and normalizes volume once at the end.

        Args:
            text_input: Text with optional control tags (e.g. <|emotion:happy|>).
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

    # ------------------------------------------------------------------
    # Public API: streaming
    # ------------------------------------------------------------------

    def stream(self, text_input, voice="alloy", chunk_size=200,
               temperature=None, top_k=None, top_p=None, seed=None,
               max_steps=None, reference_audio=None, reference_text=None,
               add_preroll=True, close_utterance=True, decode_every=50):
        """Incremental streaming: text -> PCM16LE byte generator.

        Auto-splits long text into segments. For each segment, yields PCM
        bytes progressively as the AR loop generates codebook rows.

        Within each segment, audio is decoded every `decode_every` AR rows
        and yielded with crossfade overlap to prevent clicks at boundaries.

        Between segments, the tail of the previous segment is crossfaded
        into the head of the next segment for seamless transitions.

        Args:
            text_input: Text to synthesize.
            voice: Voice preset name.
            chunk_size: PCM bytes per yield (default 200 = ~4ms audio).
            temperature, top_k, top_p, seed: Sampling config overrides.
            max_steps: AR step cap. None = auto.
            reference_audio: Pre-encoded delayed_ref tensor for voice cloning.
            reference_text: Transcript of reference audio.
            add_preroll: Add prosody pause before first token.
            close_utterance: Ensure terminal punctuation.
            decode_every: Yield audio every N AR rows (default 50 = ~2s).

        Yields:
            bytes: PCM16LE audio chunks (24kHz, mono).
        """
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        segments = _split_text(text_input)
        if len(segments) > 1:
            logger.info("Streaming %d segments", len(segments))

        # Pass tail between segments for crossfade at boundaries
        prev_tail = None
        for i, segment in enumerate(segments):
            if len(segments) > 1:
                logger.info("Streaming segment %d/%d", i+1, len(segments))
            steps = _compute_max_steps(segment) if max_steps is None else max_steps
            # Use a mutable container so _stream_incremental can store the tail
            tail_holder = [None]
            for pcm in self._stream_incremental(segment, cfg, steps, chunk_size,
                                                 reference_audio, reference_text,
                                                 add_preroll, close_utterance,
                                                 decode_every, prev_tail=prev_tail,
                                                 tail_out=tail_holder):
                yield pcm
            # Always reset: final batch already emitted everything including any tail
            prev_tail = None
            self._clean_vram()

    def stream_from_tokens(self, text_token_generator, voice="alloy",
                           chunk_size=200, temperature=None, top_k=None,
                           top_p=None, seed=None, reference_audio=None,
                           reference_text=None, max_buffer_sec=5.0, decode_every=50):
        """LLM token pipeline: text tokens -> PCM audio stream.

        Buffers incoming tokens from an LLM, detects sentence boundaries,
        and synthesizes each complete utterance with incremental streaming.

        This is the right choice for real-time chatbot TTS where text
        arrives token-by-token from an LLM.

        Args:
            text_token_generator: Generator yielding text tokens (str).
            voice, temperature, top_k, top_p, seed: Sampling config.
            chunk_size: PCM bytes per yield.
            reference_audio: Pre-encoded delayed_ref tensor.
            reference_text: Transcript of reference audio.
            max_buffer_sec: Max buffer time before force-flush.
            decode_every: AR rows between incremental yields.

        Yields:
            bytes: PCM16LE audio chunks (24kHz, mono).
        """
        cfg = self._resolve_config(voice, temperature, top_k, top_p, seed)
        sentence_re = re.compile(r'([.!?。\u203f！\u203f]+[\s"]*)')
        buffer = ""

        def flush(text):
            """Synthesize one utterance and yield PCM."""
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
                # Force-flush if buffer gets too long (avoid excessive latency)
                if _estimate_text_seconds(buffer) >= max_buffer_sec and buffer.strip():
                    yield from flush(buffer)
                    buffer = ""
                # Flush at sentence boundaries
                match = sentence_re.search(buffer)
                if match:
                    utterance = buffer[:match.end()]
                    buffer = buffer[match.end():]
                    yield from flush(utterance)
        finally:
            # Flush remaining text on generator exit
            if buffer.strip():
                logger.info("Pipeline flushing: %s", buffer[:80])
                yield from flush(buffer)

    # ------------------------------------------------------------------
    # Internal: config resolution
    # ------------------------------------------------------------------

    def _resolve_config(self, voice, temperature, top_k, top_p, seed):
        """Merge voice preset defaults with explicit parameter overrides."""
        preset = VOICE_PRESETS.get(voice, VOICE_PRESETS["alloy"])
        return GenerationConfig(
            temperature=temperature if temperature is not None else preset["temperature"],
            top_k=top_k if top_k is not None else preset["top_k"],
            top_p=top_p if top_p is not None else preset["top_p"],
            seed=seed if seed is not None else preset["seed"],
        )

    # ------------------------------------------------------------------
    # Internal: full AR generation (for batch synthesis)
    # ------------------------------------------------------------------

    def _run_ar_generation(self, text_input, cfg, max_steps,
                            reference_audio, reference_text,
                            add_preroll, close_utterance):
        """Run the full AR generation loop. Returns list of codebook rows.

        Each row is a [N_codebooks] tensor of sampled token IDs.
        The rows are in "delayed" format (codebook c offset by c steps).
        """
        N = self.num_codebooks
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        # Handle reference audio (voice cloning)
        delayed_ref = None
        if reference_audio is not None:
            delayed_ref = (reference_audio if isinstance(reference_audio, torch.Tensor)
                          else self.encode_reference_audio(reference_audio))

        # Build prompt: preprocess text + tokenize
        source = (_ensure_terminal_punctuation(text_input) if close_utterance
                  else text_input.strip())
        gen_text = (_add_generation_preroll(source) if add_preroll else source)
        num_ref = 0 if delayed_ref is None else delayed_ref.shape[0]
        prompt_ids = self.model._build_prompt_ids(
            self.tokenizer, gen_text, num_ref_tokens=num_ref, reference_text=reference_text)

        state = _SamplerState(N)
        rows = []

        with torch.inference_mode():
            # Prefill: embed the text prompt (+ reference if any)
            embeds = self.model._prefill_embeds(prompt_ids, delayed_ref)
            out = self.model.model(inputs_embeds=embeds, use_cache=True)
            past, hidden, pos = out.past_key_values, out.last_hidden_state[:, -1, :], embeds.shape[1]

            # AR loop: one codebook row per step
            for step in range(max_steps):
                logits = self.model.audio_head(hidden).to(torch.float32)[0]
                codes = _sampler_step(logits, state, cfg.temperature, cfg.top_p, cfg.top_k)
                if state.generation_done:
                    logger.debug("EOC at step %d", step)
                    break
                rows.append(codes.cpu())

                # Feed sampled codes back into the model for next step
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

    # ------------------------------------------------------------------
    # Internal: incremental streaming with crossfade
    # ------------------------------------------------------------------

    def _stream_incremental(self, text_input, cfg, max_steps, chunk_size,
                             reference_audio, reference_text,
                             add_preroll, close_utterance, decode_every,
                             prev_tail=None, tail_out=None):
        """Incremental AR generation with sample-level crossfade for gapless output.

        Architecture:
        1. AR loop generates codebook rows one-by-one
        2. Every `decode_every` rows, decode ALL accumulated rows through vocoder
        3. Track exact sample position (no frame-to-sample conversion drift)
        4. Hold back right-edge samples (vocoder unstable zone)
        5. Crossfade 10ms overlap between consecutive batches
        6. Final flush emits everything remaining with silence trimming

        The crossfade eliminates clicks at batch boundaries by blending
        the tail of the previous decode with the head of the new one.
        """
        N = self.num_codebooks
        FADE = STREAM_CROSSFADE_SAMPLES  # 10ms crossfade region
        HOLD = STREAM_HOLDBACK_SAMPLES   # right-edge holdback

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        # Reference audio setup (same as _run_ar_generation)
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
        samples_emitted = 0   # exact sample count already yielded
        # prev_tail: FADE samples from previous batch/segment for crossfade
        _prev_tail = prev_tail  # use passed-in tail (from previous segment) or None

        def _yield_pcm(segment):
            """Convert a float32 audio segment to PCM16LE bytes and yield."""
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
            """Linear crossfade: old fades out, new fades in over fade_len samples."""
            fl = min(fade_len, len(old_tail), len(new_head))
            if fl <= 0:
                return np.concatenate([old_tail, new_head])
            fade_out = np.linspace(1.0, 0.0, fl)
            fade_in = np.linspace(0.0, 1.0, fl)
            blended = old_tail[-fl:] * fade_out + new_head[:fl] * fade_in
            return np.concatenate([old_tail[:-fl], blended, new_head[fl:]])

        def _emit_batch(is_final=False):
            """Decode current rows, crossfade with previous tail, emit new samples.

            Strategy: decode ALL rows each time (full context = most stable samples).
            Track exact sample position to avoid drift. Hold back right edge.
            """
            nonlocal samples_emitted, _prev_tail

            if len(rows) < N:
                return

            # Decode ALL accumulated rows for maximum stability
            delayed_LN = torch.stack(rows, dim=0)
            codes_TN = _reverse_delay_pattern(delayed_LN)
            wav = self.model._decode_codes(codes_TN.to(self.model.device))
            wav_np = wav.numpy().astype(np.float32)
            total_samples = len(wav_np)

            if total_samples <= samples_emitted:
                del delayed_LN, codes_TN, wav, wav_np
                return

            # How far to emit: all samples (final) or minus holdback (incremental)
            emit_to = total_samples if is_final else max(samples_emitted + 1, total_samples - HOLD)

            # Extract new audio segment
            new_audio = wav_np[samples_emitted:emit_to].copy()

            # Crossfade with previous batch's tail to eliminate clicks
            if _prev_tail is not None and len(_prev_tail) > 0:
                new_audio = _crossfade(_prev_tail, new_audio, FADE)

            # Keep tail for next crossfade (unless this is the final batch)
            if not is_final and len(new_audio) > FADE:
                _prev_tail = new_audio[-FADE:].copy()
                new_audio = new_audio[:-FADE]   # don't emit the tail yet
            else:
                _prev_tail = None

            if len(new_audio) > 0:

                yield from _yield_pcm(new_audio)
                samples_emitted = emit_to

            del delayed_LN, codes_TN, wav, wav_np

        # --- AR generation loop ---
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

                # Incremental decode: yield audio every decode_every rows
                if len(rows) >= N + decode_every and (len(rows) - N) % decode_every == 0:
                    yield from _emit_batch(is_final=False)
                    logger.debug("Emitted %.1fs (step %d, rows %d)",
                                 samples_emitted / SAMPLE_RATE, step, len(rows))

                # Feed sampled codes back into the model
                step_emb = self.model.audio_embedding(codes.unsqueeze(0)).unsqueeze(1)
                cpos = torch.tensor([pos], device=self.model.device)
                out = self.model.model(inputs_embeds=step_emb.to(embeds.dtype),
                                       past_key_values=past, use_cache=True, cache_position=cpos)
                past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
                del logits, codes, step_emb, cpos
                pos += 1

            # Final flush: emit all remaining samples including holdback zone
            if rows:
                yield from _emit_batch(is_final=True)

            # Store final tail for inter-segment crossfade (if any remaining)
            if tail_out is not None:
                tail_out[0] = _prev_tail

            logger.info("Stream done: %d rows, %.1fs total audio",
                        len(rows), samples_emitted / SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Internal: batch decode + post-processing
    # ------------------------------------------------------------------

    def _decode_rows(self, rows):
        """Decode codebook rows -> processed numpy audio (normalized).

        Used by synthesize() for batch mode. Applies guard silence,
        trailing silence trim, and RMS normalization.
        """
        with torch.inference_mode():
            delayed_LN = torch.stack(rows, dim=0)
            codes_TN = _reverse_delay_pattern(delayed_LN)
            wav = self.model._decode_codes(codes_TN.to(self.model.device))
            wav_np = wav.numpy().astype(np.float32)

        # Prepend guard silence (compensates for vocoder left-edge artifacts)
        silence = np.zeros(int(POST_DECODE_SILENCE_SEC * SAMPLE_RATE), dtype=wav_np.dtype)
        wav_np = np.concatenate([silence, wav_np])
        wav_np = _trim_trailing_silence(wav_np)
        wav_np = _normalize_audio(wav_np)
        return wav_np

    def _write_wav(self, wav_np, output_path):
        """Write numpy audio array to WAV file (24kHz, mono, 16-bit)."""
        pcm = np.clip(wav_np * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

    def _clean_vram(self):
        """Aggressively reclaim VRAM after generation.

        Moves the vocoder (audio_codec) to CPU and frees cached tensors.
        Important for preventing OOM during long multi-segment synthesis.
        """
        if hasattr(self.model, "_audio_codec") and self.model._audio_codec is not None:
            self.model._audio_codec.cpu()
            self.model._audio_codec = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def get_vram_info(self):
        """Print current VRAM usage stats."""
        torch.cuda.synchronize(self._device_idx)
        free = torch.cuda.mem_get_info(self._device_idx)[0] / 1e9
        total = torch.cuda.get_device_properties(self._device_idx).total_memory / 1e9
        alloc = torch.cuda.memory_allocated(self._device_idx) / 1e6
        peak = torch.cuda.max_memory_allocated(self._device_idx) / 1e6
        logger.info("VRAM: %.1f/%.0f GB free, Alloc: %.0f MB, Peak: %.0f MB",
                     free, total, alloc, peak)
