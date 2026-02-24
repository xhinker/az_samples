#!/usr/bin/env python3
"""
text_processor.py — Ebook text → structured speech-segment list via LLM

Converts raw ebook/novel text into a JSON list where every element describes
one spoken unit (dialog line, narrative passage, or environment sound) together
with the character role, gender, and emotion metadata needed by IndexTTS.

Output segment schema
---------------------
{
    "role":    "role_name",                         # character name or "Narrator"
    "gender":  "male" | "female" | "neutral",
    "emotion": "emotion description",               # e.g. "angry", "whispered", "calm"
    "type":    "dialog" | "narrator" | "environment_sound",
    "text":    "the actual text to synthesise"
}

Role config schema (auto-generated, user fills in audio_file)
-------------------------------------------------------------
{
    "role":       "role_name",
    "gender":     "male" | "female" | "neutral",
    "audio_file": ""                                # user fills this in
}

Usage
-----
    from text_processor import TextProcessor

    tp = TextProcessor()
    segments = tp.process_text(raw_text)
    tp.save_segments(segments, "segments.json")
    tp.save_roles_config(segments, "roles_config.json")
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from openai import OpenAI

# ── LLM connection defaults ──────────────────────────────────────────────────
DEFAULT_LLM_API_URL   = "http://192.168.68.79:1234/v1"
DEFAULT_LLM_API_KEY   = "lmstudio"
DEFAULT_LLM_MODEL_NAME = "qwen/qwen3-coder-next"

# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert audiobook producer and text analyser.

Your job is to convert raw ebook text into a structured list of speech segments
suitable for text-to-speech synthesis with different voice actors per role.

For every distinct spoken unit in the source text, produce ONE JSON object:

{
  "role":    <character name, "Narrator", or "Environment">,
  "gender":  <"male" | "female" | "neutral">,
  "emotion": <one concise phrase, e.g. "calm", "angry", "excited", "whispering", "fearful">,
  "type":    <"dialog" | "narrator" | "environment_sound">,
  "text":    <the EXACT original text — do NOT paraphrase or summarise>
}

═══════════════════════════════════════════════════════
CRITICAL RULE — Chinese novel dialog-attribution split
═══════════════════════════════════════════════════════
In Chinese novels, a character's spoken words and the attribution (who said
it / how) are often written ON THE SAME LINE with NO newline between them:

    "台词内容。"角色名动词描述。

You MUST split every such line into EXACTLY TWO segments:

  Segment A — the spoken dialog:
    {"role": "角色名", "type": "dialog",   "text": ""台词内容。""}

  Segment B — the attribution (narration):
    {"role": "Narrator", "type": "narrator", "text": "角色名动词描述。"}

NEVER merge dialog text and its attribution into a single segment.

────────────────────────────────────────────────────
Chinese example — source line:
  "不如今天咱们吃虾吧！"亚男提议道。

WRONG (merged — DO NOT DO THIS):
  {"role":"亚男","type":"dialog","text":""不如今天咱们吃虾吧！"亚男提议道。"}

CORRECT (split into two):
  {"role":"亚男",    "type":"dialog",   "text":""不如今天咱们吃虾吧！""},
  {"role":"Narrator","type":"narrator", "text":"亚男提议道。"}

────────────────────────────────────────────────────
Another example — source line:
  "对了，小曼，你不是说有事跟大家说吗？什么事？"傅蕾问道。

CORRECT (split):
  {"role":"傅蕾",    "type":"dialog",   "text":""对了，小曼，你不是说有事跟大家说吗？什么事？""},
  {"role":"Narrator","type":"narrator", "text":"傅蕾问道。"}

────────────────────────────────────────────────────
Another example with longer attribution — source line:
  "是啊，国际国内连续倒班，根本倒不过来，我都累趴下了。"亚男一副苦大仇深的样子。

CORRECT (split):
  {"role":"亚男",    "type":"dialog",   "text":""是啊，国际国内连续倒班，根本倒不过来，我都累趴下了。""},
  {"role":"Narrator","type":"narrator", "text":"亚男一副苦大仇深的样子。"}

════════════════════════════════════════════════════

General rules
─────────────
1. "role":
   - Use the CHARACTER'S NAME for lines they speak (infer from dialogue attribution).
   - Use "Narrator" for all descriptive / narrative passages.
   - Use "Environment" for action / sound descriptions (e.g. *thunder crashes*,
     [a door slams], stage directions in italics/brackets).

2. "gender":
   - Infer from pronouns, names, or context clues.
   - Use "neutral" when genuinely unknown.

3. "emotion":
   - Describe the speaking tone in 1-4 words. Examples: "cheerful", "cold and distant",
     "barely above a whisper", "sarcastic", "matter-of-fact".

4. "type":
   - "dialog"            – a character's spoken words.
   - "narrator"          – narrative / descriptive text.
   - "environment_sound" – ambient sounds, action lines, stage directions.

5. "text":
   - Copy the EXACT source text. Do not alter punctuation, spelling, or wording.
   - Include surrounding quotation marks for dialog if they appear in the source.

6. Split ONLY at natural role / type boundaries. Do not merge separate characters'
   lines into one segment. Do not split a single continuous sentence mid-way.

7. Return ONLY a valid JSON array — no markdown fences, no commentary, no extra keys.

Good general example
────────────────────
[
  {"role":"Narrator","gender":"neutral","emotion":"tense","type":"narrator",
   "text":"The tavern door swung open with a crash."},
  {"role":"Environment","gender":"neutral","emotion":"dramatic","type":"environment_sound",
   "text":"*Wind howled through the gap.*"},
  {"role":"Innkeeper","gender":"male","emotion":"irritated","type":"dialog",
   "text":"We're closed! he barked."},
  {"role":"Lyra","gender":"female","emotion":"confident","type":"dialog",
   "text":"Not for me, you're not."}
]
"""

USER_PROMPT_TMPL = """\
Convert the following ebook text into speech segments.
Return ONLY the JSON array — nothing else.

--- BEGIN TEXT ---
{text}
--- END TEXT ---
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_llm_wrapper(raw: str) -> str:
    """Remove <think>…</think> reasoning blocks and markdown code fences."""
    # Remove <think> blocks (Qwen3 chain-of-thought)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Remove ``` fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```", "", raw)
    return raw.strip()


def _normalize_segment(seg: dict[str, Any]) -> dict[str, str]:
    """Normalise one raw LLM segment dict to the canonical schema."""
    # The user's spec has "emotion:" (with a stray colon) as the key name;
    # handle both forms gracefully.
    emotion = seg.get("emotion") or seg.get("emotion:") or "neutral"
    return {
        "role":    str(seg.get("role",   "Narrator")).strip(),
        "gender":  str(seg.get("gender", "neutral")).strip().lower(),
        "emotion": str(emotion).strip(),
        "type":    str(seg.get("type",   "narrator")).strip().lower(),
        "text":    str(seg.get("text",   "")).strip(),
    }


# ── Post-processing: split dialog + attribution ───────────────────────────────

# Chinese opening/closing quotation marks used in novels
_OPEN_QUOTES  = "\u201c\u300c\u300e"   # " 「 『
_CLOSE_QUOTES = "\u201d\u300d\u300f"   # " 」 』

# Pattern: a segment whose text is exactly  "dialog"attribution
# Group 1 = the fully-quoted dialog (opening quote … closing quote)
# Group 2 = the attribution text after the closing quote (≥ 2 chars)
_DIALOG_ATTR_RE = re.compile(
    r"^([" + re.escape(_OPEN_QUOTES) + r"]"      # opening quote
    r"[^" + re.escape(_CLOSE_QUOTES) + r"]*"     # dialog content (no closing quote inside)
    r"[" + re.escape(_CLOSE_QUOTES) + r"])"      # closing quote
    r"(\S.+)$",                                   # attribution (starts with non-whitespace)
    re.DOTALL,
)


def _split_dialog_attribution(
    segment: dict[str, str],
    known_roles: dict[str, str],   # role_name → gender
) -> list[dict[str, str]]:
    """
    If *segment*'s text matches the Chinese pattern "dialog"attribution,
    return two segments: a dialog segment and a narrator attribution segment.

    The speaker is determined from:
      1. segment["role"] if it is already a named character (not Narrator/Environment), OR
      2. Scanning the start of the attribution text for a known character name.

    Returns the original segment unchanged if the pattern does not match or
    the speaker cannot be determined.
    """
    text = segment.get("text", "").strip()
    m = _DIALOG_ATTR_RE.match(text)
    if not m:
        return [segment]

    dialog_part = m.group(1).strip()
    attr_part   = m.group(2).strip()

    # Attribution must be substantial enough to be its own segment
    if len(attr_part) < 2:
        return [segment]

    seg_role    = segment.get("role", "Narrator")
    seg_gender  = segment.get("gender", "neutral")
    seg_emotion = segment.get("emotion", "neutral")

    # Case A: LLM already assigned a named character — split cleanly
    if seg_role not in ("Narrator", "Environment"):
        speaker_role   = seg_role
        speaker_gender = seg_gender
    else:
        # Case B: LLM assigned Narrator — try to find the speaker name at the
        # start of the attribution text by matching against known character names.
        # Sort by length descending so longer names are tried first (避免前缀误匹配).
        speaker_role   = None
        speaker_gender = "neutral"
        for role in sorted(known_roles, key=len, reverse=True):
            if attr_part.startswith(role):
                speaker_role   = role
                speaker_gender = known_roles[role]
                break

        if speaker_role is None:
            # Cannot determine speaker — leave segment untouched
            return [segment]

    return [
        {
            "role":    speaker_role,
            "gender":  speaker_gender,
            "emotion": seg_emotion,
            "type":    "dialog",
            "text":    dialog_part,
        },
        {
            "role":    "Narrator",
            "gender":  "neutral",
            "emotion": "neutral",
            "type":    "narrator",
            "text":    attr_part,
        },
    ]


# ── Main class ────────────────────────────────────────────────────────────────

class TextProcessor:
    """
    Convert raw ebook text to structured speech-segment JSON using an LLM.

    Parameters
    ----------
    api_url : str
        Base URL of an OpenAI-compatible chat-completion endpoint.
    api_key : str
        API key (can be a dummy value for local LM Studio servers).
    model_name : str
        Model identifier sent to the API.
    chunk_size : int
        Approximate character limit per LLM call.  Large texts are split at
        paragraph boundaries so each call stays within the model's context.
    """

    def __init__(
        self,
        api_url:    str = DEFAULT_LLM_API_URL,
        api_key:    str = DEFAULT_LLM_API_KEY,
        model_name: str = DEFAULT_LLM_MODEL_NAME,
        chunk_size: int = 3000,
    ) -> None:
        self.client     = OpenAI(base_url=api_url, api_key=api_key)
        self.model_name = model_name
        self.chunk_size = chunk_size

    # ── Public API ─────────────────────────────────────────────────────────

    def process_text(self, text: str) -> list[dict[str, str]]:
        """
        Segment *text* into speech units.

        Steps
        -----
        1. Split large text into paragraph-boundary chunks.
        2. Send each chunk to the LLM and collect normalised segments.
        3. Post-process: split any "dialog"attribution combos the LLM missed.

        Returns
        -------
        list[dict]
            List of normalized segment dicts (role / gender / emotion / type / text).
        """
        chunks = self._split_into_chunks(text)
        total  = len(chunks)
        all_segments: list[dict[str, str]] = []

        for i, chunk in enumerate(chunks, start=1):
            print(f"  Processing chunk {i}/{total} ({len(chunk)} chars)…", flush=True)
            segments = self._call_llm(chunk)
            all_segments.extend(segments)

        # Build a role→gender map from all LLM-assigned named characters so
        # the post-processor can identify speakers from attribution text.
        known_roles: dict[str, str] = {
            seg["role"]: seg.get("gender", "neutral")
            for seg in all_segments
            if seg.get("role") not in ("Narrator", "Environment", "")
        }

        # Post-process: split "dialog"attribution segments the LLM merged
        post_processed: list[dict[str, str]] = []
        split_count = 0
        for seg in all_segments:
            parts = _split_dialog_attribution(seg, known_roles)
            post_processed.extend(parts)
            if len(parts) > 1:
                split_count += 1

        if split_count:
            print(f"  Post-processed: split {split_count} dialog+attribution segment(s).")

        print(f"  Done. {len(post_processed)} segments total.")
        return post_processed

    def save_segments(self, segments: list[dict], path: str) -> None:
        """Write the segment list to *path* as pretty-printed JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(segments)} segments → {path}")

    def extract_roles(self, segments: list[dict]) -> list[dict[str, str]]:
        """
        Derive the unique role list from *segments*.

        Returns one entry per role, preserving first-seen gender.
        The ``audio_file`` field is left blank for the user to fill in.
        """
        seen: dict[str, dict] = {}
        for seg in segments:
            role = seg["role"]
            if role not in seen:
                seen[role] = {
                    "role":       role,
                    "gender":     seg.get("gender", "neutral"),
                    "audio_file": "",
                }
        return list(seen.values())

    def save_roles_config(self, segments: list[dict], path: str) -> None:
        """Write the roles configuration skeleton to *path*."""
        roles = self.extract_roles(segments)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(roles, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(roles)} roles → {path}")
        print("NOTE: Fill in the 'audio_file' paths in the roles config before generating audio.")

    # ── Internal ────────────────────────────────────────────────────────────

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split *text* at paragraph boundaries keeping each chunk ≤ chunk_size chars."""
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _call_llm(self, text: str) -> list[dict[str, str]]:
        """Send one chunk to the LLM and return the parsed segment list."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_TMPL.format(text=text)},
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content or ""
        raw = _strip_llm_wrapper(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Attempt to salvage a partial JSON array
            match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                raise ValueError(f"LLM returned non-JSON output:\n{raw[:500]}") from exc

        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")

        return [_normalize_segment(seg) for seg in parsed if isinstance(seg, dict)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ebook text to TTS speech-segment JSON via LLM.",
    )
    parser.add_argument("input",  help="Path to the input text file")
    parser.add_argument(
        "-o", "--output", default="segments.json",
        help="Path for the output segments JSON (default: segments.json)",
    )
    parser.add_argument(
        "-r", "--roles", default="roles_config.json",
        help="Path for the roles configuration JSON (default: roles_config.json)",
    )
    parser.add_argument("--api-url",   default=DEFAULT_LLM_API_URL)
    parser.add_argument("--api-key",   default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--model",     default=DEFAULT_LLM_MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=3000)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    tp = TextProcessor(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        chunk_size=args.chunk_size,
    )

    print(f"Processing: {args.input}  ({len(text)} chars)")
    segments = tp.process_text(text)
    tp.save_segments(segments, args.output)
    tp.save_roles_config(segments, args.roles)


if __name__ == "__main__":
    main()
