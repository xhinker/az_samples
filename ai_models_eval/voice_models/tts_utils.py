#!/usr/bin/env python3
"""TTS utilities: text chunking (bilingual EN/CN) and WAV concatenation."""

import re
import wave
import numpy as np
from pathlib import Path

# --- Text chunking constants ---
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SPLIT_HINT_CHARS = ",;，；。！？!?、—-"  # NO space — never break mid-phrase
TEXT_REANCHOR_MAX_WORDS = 90
TEXT_REANCHOR_TARGET_SECONDS = 30.0


def split_text_for_reanchor(
    text: str,
    max_words: int = TEXT_REANCHOR_MAX_WORDS,
    target_seconds: float = TEXT_REANCHOR_TARGET_SECONDS,
) -> list[str]:
    """Split long text (English + Chinese) into chunks for TTS inference.

    Strategy:
    1. Split on sentence boundaries (.!?。！？)
    2. Estimate audio duration per piece (different rates for CJK vs Latin)
    3. Merge sentences until target_seconds is reached
    4. If a single sentence is too long, split at nearest punctuation

    Args:
        text: Input text (English, Chinese, or mixed).
        max_words: Hard cap on words per segment.
        target_seconds: Target audio duration per segment.

    Returns:
        List of text segments, each suitable for a single TTS inference call.
    """
    clean = " ".join(text.strip().split())
    if not clean:
        return []

    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", clean) if s]
    segments: list[str] = []

    def estimate_seconds(piece: str) -> float:
        normalized = " ".join(piece.strip().split())
        if not normalized:
            return 0.0
        non_space_chars = len(normalized.replace(" ", ""))
        words = len(normalized.split())
        cjk_chars = len(_CJK_CHAR_RE.findall(normalized))
        if cjk_chars >= max(8, int(non_space_chars * 0.20)):
            # CJK speech is better approximated by character count.
            return (cjk_chars / 4.0) + ((non_space_chars - cjk_chars) / 12.0)
        return max(words / 2.5, non_space_chars / 12.0)

    def split_oversized_piece(piece: str) -> list[str]:
        normalized = " ".join(piece.strip().split())
        if not normalized:
            return []
        non_space = len(normalized.replace(" ", ""))
        if estimate_seconds(normalized) <= target_seconds and non_space <= 120:
            return [normalized]

        non_space_chars = len(normalized.replace(" ", ""))
        cjk_chars = len(_CJK_CHAR_RE.findall(normalized))
        is_cjk_heavy = cjk_chars >= max(8, int(non_space_chars * 0.20))
        chars_per_second = 4 if is_cjk_heavy else 12
        window = min(120, max(80, int(target_seconds * chars_per_second)))

        parts: list[str] = []
        start = 0
        text_len = len(normalized)
        while start < text_len:
            end = min(text_len, start + window)
            if end < text_len:
                cut = -1
                scan_start = max(start + int(window * 0.5), start + 1)
                for idx in range(end, scan_start - 1, -1):
                    if normalized[idx - 1] in _SPLIT_HINT_CHARS:
                        cut = idx
                        break
                if cut == -1:
                    cut = end
            else:
                cut = end
            part = normalized[start:cut].strip()
            if part:
                parts.append(part)
            start = cut
        return parts

    current = ""

    def flush_current():
        nonlocal current
        if current:
            segments.append(current)
            current = ""

    for sentence in sentences:
        for piece in split_oversized_piece(sentence):
            words = len(piece.split())
            if words > max_words:
                flush_current()
                start = 0
                word_list = piece.split()
                while start < len(word_list):
                    part_words = word_list[start:start + max_words]
                    segments.append(" ".join(part_words))
                    start += max_words
                continue

            candidate = piece if not current else f"{current} {piece}"
            candidate_chars = len(candidate.replace(" ", ""))
            # Flush if: exceeds time target OR exceeds hard char limit (safety net)
            if (current and estimate_seconds(candidate) > target_seconds) or candidate_chars > 120:
                flush_current()
                current = piece
            else:
                current = candidate

    flush_current()
    return segments if segments else [clean]


def concatenate_wavs(wav_paths: list[str], output_path: str, sample_rate: int = 24000) -> str:
    """Concatenate multiple WAV files into one.

    All input WAVs must have the same sample rate and be mono 16-bit PCM.
    Silently skips empty or unreadable files.

    Args:
        wav_paths: List of paths to WAV files to concatenate.
        output_path: Path for the output WAV file.
        sample_rate: Expected sample rate (for validation).

    Returns:
        Path to the concatenated WAV file.
    """
    all_frames = []
    for wp in wav_paths:
        p = Path(wp)
        if not p.exists():
            continue
        try:
            with wave.open(str(p), "rb") as wf:
                if wf.getframerate() != sample_rate:
                    print(f"  WARNING: {p.name} has sample rate {wf.getframerate()}, expected {sample_rate}. Skipping.")
                    continue
                nframes = wf.getnframes()
                if nframes == 0:
                    continue
                raw = wf.readframes(nframes)
                all_frames.append(raw)
        except Exception as e:
            print(f"  WARNING: Could not read {p.name}: {e}")

    if not all_frames:
        raise ValueError("No valid WAV frames to concatenate.")

    combined = b"".join(all_frames)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(combined)

    duration = len(combined) / 2 / sample_rate
    print(f"Concatenated {len(all_frames)} WAV(s) -> {output_path} ({duration:.1f}s)")
    return output_path
