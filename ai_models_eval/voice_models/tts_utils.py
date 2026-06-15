#!/usr/bin/env python3
"""TTS utilities: text chunking (bilingual EN/CN) and WAV concatenation."""

import re
import wave
import numpy as np
from pathlib import Path

# --- Text chunking constants ---
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TERMINAL_PUNCT_CHARS = ".!?。！？"
_WEAK_PUNCT_CHARS = ",;，；、:："
_SPLIT_HINT_CHARS = _TERMINAL_PUNCT_CHARS + _WEAK_PUNCT_CHARS + "—-"  # NO space — never break mid-phrase
_CLOSING_QUOTE_CHARS = "\"'”’）)]》」』"
TEXT_REANCHOR_MAX_WORDS = 28
TEXT_REANCHOR_TARGET_SECONDS = 12.0


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

    sentence_pattern = rf"[^.!?。！？]+(?:[.!?。！？]+[{re.escape(_CLOSING_QUOTE_CHARS)}]*)?"
    sentences = [m.group(0).strip() for m in re.finditer(sentence_pattern, clean) if m.group(0).strip()]
    segments: list[str] = []

    def is_cjk_heavy(piece: str) -> bool:
        """Determine if a text piece is predominantly CJK (Chinese/Japanese/Korean).

        Used to pick the right speech rate: CJK is timed by character count,
        Latin by word count. Mixed text like "Chapter 1: 今天天气很好" falls
        back to Latin timing because English words dilute the CJK ratio.

        Criteria: CJK chars must be >= 8 AND >= 20% of all non-space chars.
        The absolute minimum (8) avoids misclassifying short fragments like
        "你好世界" (4 chars) as CJK-heavy when they're too short to tell.
        The 20% ratio handles mixed text: "Hello 你好" = 6 CJK of 11 total
        = 54% but only 6 chars, so fails the absolute minimum.
        """
        non_space_chars = len(piece.replace(" ", ""))
        cjk_chars = len(_CJK_CHAR_RE.findall(piece))
        return cjk_chars >= max(8, int(non_space_chars * 0.20))

    def segment_terminal(piece: str) -> str:
        """Return the appropriate terminal punctuation for the text's language.

        Used by close_tts_segment() when a chunk needs a closing period added.
        Chinese text gets "。", English/Latin gets ".".

        Without this, mixed-language chunks would get the wrong punctuation:
        "你过得好吗" + "." → "你过得好吗." (wrong)
        "How are you" + "。" → "How are you。" (wrong)
        """
        return "。" if is_cjk_heavy(piece) else "."

    def close_tts_segment(piece: str) -> str:
        """Make each chunk sound like a complete utterance.

        Higgs often keeps generating when the text ends with weak punctuation
        such as a comma because that means "continue" rather than "stop".
        """
        # Before: " Hello\t\tworld\n\n " After: "Hello world"
        normalized = " ".join(piece.strip().split())
        if not normalized:
            return ""

        suffix = ""
        while normalized and normalized[-1] in _CLOSING_QUOTE_CHARS:
            suffix = normalized[-1] + suffix
            normalized = normalized[:-1].rstrip()

        if not normalized:
            return suffix
        if normalized[-1] in _TERMINAL_PUNCT_CHARS:
            return normalized + suffix
        if normalized[-1] in _WEAK_PUNCT_CHARS:
            normalized = normalized[:-1].rstrip()
        return normalized + segment_terminal(normalized) + suffix

    def estimate_seconds(piece: str) -> float:
        normalized = " ".join(piece.strip().split())
        if not normalized:
            return 0.0
        non_space_chars = len(normalized.replace(" ", ""))
        words = len(normalized.split())
        cjk_chars = len(_CJK_CHAR_RE.findall(normalized))
        if is_cjk_heavy(normalized):
            # CJK speech is better approximated by character count.
            return (cjk_chars / 4.0) + ((non_space_chars - cjk_chars) / 12.0)
        return max(words / 1.8, non_space_chars / 10.0)

    def split_oversized_piece(piece: str) -> list[str]:
        normalized = " ".join(piece.strip().split())
        if not normalized:
            return []
        non_space = len(normalized.replace(" ", ""))
        if estimate_seconds(normalized) <= target_seconds and non_space <= 120:
            return [normalized]

        non_space_chars = len(normalized.replace(" ", ""))
        cjk_heavy = is_cjk_heavy(normalized)
        words = normalized.split()
        if not cjk_heavy and len(words) > max_words:
            group_count = max(1, (len(words) + max_words - 1) // max_words)
            group_size = max(1, (len(words) + group_count - 1) // group_count)
            return [
                " ".join(words[start : start + group_size])
                for start in range(0, len(words), group_size)
            ]

        chars_per_second = 4 if cjk_heavy else 10
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
                    if normalized[idx - 1] in _TERMINAL_PUNCT_CHARS:
                        cut = idx
                        break
                if cut == -1:
                    for idx in range(end, scan_start - 1, -1):
                        if normalized[idx - 1] in _WEAK_PUNCT_CHARS:
                            cut = idx
                            break
                if cut == -1:
                    for idx in range(end, scan_start - 1, -1):
                        if normalized[idx - 1] in "—-":
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
            candidate_words = len(candidate.split())
            # Flush if: exceeds time target OR exceeds hard limits (safety net)
            if (
                current
                and (
                    estimate_seconds(candidate) > target_seconds
                    or candidate_chars > 120
                    or candidate_words > max_words
                )
            ):
                flush_current()
                current = piece
            else:
                current = candidate

    flush_current()
    segments = [closed for segment in segments if (closed := close_tts_segment(segment))]
    return segments if segments else [close_tts_segment(clean)]


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
