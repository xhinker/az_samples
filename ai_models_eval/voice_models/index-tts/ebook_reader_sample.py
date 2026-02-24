#!/usr/bin/env python3
"""
ebook_reader_sample.py — End-to-end ebook reader demo
======================================================

Pipeline
--------
1. Use TextProcessor (LLM) to convert a sample text passage into a structured
   list of speech segments (segments.json).
2. Auto-generate a role audio configuration file (roles_config.json).
3. Register each role's reference audio with the IndexTTS API's /v1/voices
   endpoint (skipped if no audio_file is set for a role).
4. For each segment, call the IndexTTS API to synthesise audio and save it
   under output/<index>_<role>_<type>.wav.

Prerequisites
-------------
- IndexTTS API must be running on INDEXTTS_API_URL (default http://localhost:8085).
  Start it with:  ./indextts-venv-py311/bin/python indextts-api.py
- Fill in the "audio_file" paths in roles_config.json before step 4, OR run
  this script to process text first (step 1-2), then edit roles_config.json,
  then re-run with --skip-process to jump straight to audio generation.

Usage
-----
  # Full pipeline (process text + generate audio):
  ./indextts-venv-py311/bin/python ebook_reader_sample.py

  # Only text processing (no audio generation):
  ./indextts-venv-py311/bin/python ebook_reader_sample.py --text-only

  # Only audio generation (use existing segments.json / roles_config.json):
  ./indextts-venv-py311/bin/python ebook_reader_sample.py --audio-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
SEGMENTS_FILE    = os.path.join(BASE_DIR, "segments.json")
ROLES_CONFIG     = os.path.join(BASE_DIR, "roles_config.json")
OUTPUT_DIR       = os.path.join(BASE_DIR, "output_audio")

# ── IndexTTS API ──────────────────────────────────────────────────────────────
INDEXTTS_API_URL = "http://localhost:8085"
DEFAULT_VOICE    = "default"          # fallback if a role has no audio_file

# ── Sample text ───────────────────────────────────────────────────────────────
# SAMPLE_TEXT = """\
# The autumn rain had not let up since morning. Detective Sarah Kane stood at the
# window of the precinct, her coffee going cold on the desk behind her.

# "You're staring again," said Officer Liang from across the room, not looking up
# from his paperwork.

# "Thinking," she corrected. "There's a difference."

# *A phone rang somewhere down the hall.*

# He finally glanced over his shoulder. "The Harmon case?"

# "Who else?" She turned from the window. The fluorescent light made her look
# tired. "Three witnesses, two contradictory timelines, and a suspect who claims
# he was forty miles away."

# Liang leaned back in his chair and laced his fingers behind his head.
# "Maybe he was."

# "Maybe." Kane picked up her coffee, found it cold, and set it back down.
# "Or maybe he had help."

# *Outside, a siren wailed and faded into the grey city noise.*

# "That's the part that worries me," she added quietly.
# """

SAMPLE_TEXT = """\
第12章 黎男的秘密
“你知道吗？我曾经幻想过那种黑道般的日子。男朋友是个黑社会的小头目，而我则成天跟在他身边，全身黑色皮衣、皮裤，骑着摩托车，在大街小巷窜来窜去。逢人便被称呼什么什么嫂啊的，比如男朋友是豹哥，我就是豹嫂。想想，多带劲啊！”
古曼走了
时间，跑得太快了，以至于一不留神，它便从你身边一晃而过，不留一点痕迹。一眨眼，几个月又过去了。
这天下午，我正在宿舍洗衣服，忽然，电话响了，是古曼打来的。
“若狐，晚上出来吃饭，我已经约好大家了，除了小柠子有任务，其他姐妹都在。”
小柠子，不知从何时起，姐妹们都这么叫木柠。
“好啊，我正洗衣服呢，一会儿我就去你宿舍找你。”
挂了电话，我看了看时间，五点一刻。黎男不在，执行航班任务去了。
洗完衣服，我简单地收拾了一下，便去了古曼的宿舍，正好赶上姐妹们都在门口。
“走吧，咱们出去吃，边吃边聊。”
不知不觉，大家逛到了一家叫“爬爬虾”的饭馆门口。
“不如今天咱们吃虾吧！”亚男提议道。
姐妹都点了点头，据说这家店的味道不错，从现在就宾客盈门的情况看来，味道应该不会差。我们选了二楼一个靠窗的角落处，坐了下来。
“亚男，飞国际很累吧？”桑影首先打开了话匣子。
这阵子大家都很忙，几乎连照面也很困难。
“是啊，国际国内连续倒班，根本倒不过来，我都累趴下了，只想公司能好好地放我几天假，让我睡个饱。”亚男一副苦大仇深的样子。
“你做梦吧，想放假，等明年的年休假吧。”傅蕾挖苦道。
“这狗屁破公司，就算是机器也该有休息的时候，还真不把我们当人了。”柳茹破口大骂起来。
骂，确实该骂。对于我们，能做的，也只能如此骂骂，发泄一下而已。
大家默默吃着饭。
这时，傅蕾突然想起了什么。
“对了，小曼，你不是说有事跟大家说吗？什么事？”傅蕾问道。
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 + 2 — Text processing
# ═══════════════════════════════════════════════════════════════════════════════

def run_text_processing() -> list[dict]:
    """Run TextProcessor on SAMPLE_TEXT and return the segment list."""
    from text_processor import TextProcessor   # local import to keep CLI fast

    print("─" * 60)
    print("Step 1/2 — Text processing")
    print("─" * 60)

    tp = TextProcessor()
    segments = tp.process_text(SAMPLE_TEXT)
    tp.save_segments(segments, SEGMENTS_FILE)
    tp.save_roles_config(segments, ROLES_CONFIG)
    return segments


def load_segments() -> list[dict]:
    with open(SEGMENTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_roles_config() -> dict[str, dict]:
    """Load roles_config.json and return a role-name → config dict."""
    if not os.path.exists(ROLES_CONFIG):
        return {}
    with open(ROLES_CONFIG, encoding="utf-8") as f:
        roles = json.load(f)
    return {r["role"]: r for r in roles}


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Register voices
# ═══════════════════════════════════════════════════════════════════════════════

async def register_voices(
    session: aiohttp.ClientSession,
    roles: dict[str, dict],
) -> dict[str, str]:
    """
    Upload each role's audio_file to the IndexTTS API as a named voice.

    Returns
    -------
    dict[role_name, voice_id]
        Maps role names to the voice identifier to use in /v1/audio/speech calls.
    """
    voice_map: dict[str, str] = {}

    for role_name, cfg in roles.items():
        audio_file = cfg.get("audio_file", "").strip()

        if not audio_file:
            print(f"  [Skip] '{role_name}' — no audio_file set; will use '{DEFAULT_VOICE}'")
            voice_map[role_name] = DEFAULT_VOICE
            continue

        if not os.path.exists(audio_file):
            print(f"  [Warn] '{role_name}' — audio_file not found: {audio_file}")
            voice_map[role_name] = DEFAULT_VOICE
            continue

        # Sanitise role name to a safe voice ID
        voice_id = "".join(c for c in role_name.lower() if c.isalnum() or c in "-_")

        print(f"  Registering '{role_name}' as voice '{voice_id}' …", end=" ", flush=True)
        with open(audio_file, "rb") as fh:
            audio_bytes = fh.read()

        form = aiohttp.FormData()
        form.add_field("name", voice_id)
        form.add_field(
            "file",
            audio_bytes,
            filename=os.path.basename(audio_file),
            content_type="audio/wav",
        )

        try:
            async with session.post(
                f"{INDEXTTS_API_URL}/v1/voices", data=form
            ) as resp:
                if resp.status in (200, 201):
                    print("OK")
                    voice_map[role_name] = voice_id
                else:
                    text = await resp.text()
                    print(f"FAILED ({resp.status}): {text[:120]}")
                    voice_map[role_name] = DEFAULT_VOICE
        except aiohttp.ClientError as exc:
            print(f"ERROR: {exc}")
            voice_map[role_name] = DEFAULT_VOICE

    return voice_map


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Generate audio for each segment
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_segment_audio(
    session:   aiohttp.ClientSession,
    segment:   dict,
    voice_map: dict[str, str],
    index:     int,
    out_dir:   str,
) -> str | None:
    """
    Call IndexTTS API for one segment and save the resulting WAV.

    Returns the output file path, or None if skipped / failed.
    """
    seg_type = segment.get("type", "narrator")
    text     = segment.get("text", "").strip()
    role     = segment.get("role", "Narrator")

    if not text:
        return None

    # Environment sounds are not synthesised as speech
    if seg_type == "environment_sound":
        print(f"  [{index:04d}] [env]  {text[:70]}")
        return None

    voice_id = voice_map.get(role, DEFAULT_VOICE)
    safe_role = "".join(c if c.isalnum() or c in "-_" else "_" for c in role)
    out_path  = os.path.join(out_dir, f"{index:04d}_{safe_role}_{seg_type}.wav")

    emotion = segment.get("emotion", "")
    label   = f"[{seg_type[:3]}]"
    print(f"  [{index:04d}] {label} {role} ({emotion}): {text[:65]}…")

    payload = {
        "model": "indextts-2",
        "input": text,
        "voice": voice_id,
    }

    try:
        async with session.post(
            f"{INDEXTTS_API_URL}/v1/audio/speech",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                print(f"        ERROR {resp.status}: {err[:100]}")
                return None
            wav_bytes = await resp.read()

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(wav_bytes)
        return out_path

    except aiohttp.ClientError as exc:
        print(f"        CLIENT ERROR: {exc}")
        return None


async def generate_all_audio(
    segments:  list[dict],
    voice_map: dict[str, str],
    out_dir:   str,
) -> list[str | None]:
    """Sequentially generate audio for every segment (avoids GPU contention)."""
    results: list[str | None] = []
    async with aiohttp.ClientSession() as session:
        for i, seg in enumerate(segments):
            path = await generate_segment_audio(session, seg, voice_map, i, out_dir)
            results.append(path)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

async def main_async(text_only: bool, audio_only: bool) -> None:
    # ── Step 1-2: text processing ───────────────────────────────────────────
    if not audio_only:
        segments = run_text_processing()
    else:
        if not os.path.exists(SEGMENTS_FILE):
            sys.exit(f"segments.json not found at {SEGMENTS_FILE}. Run without --audio-only first.")
        segments = load_segments()
        print(f"Loaded {len(segments)} segments from {SEGMENTS_FILE}")

    if text_only:
        print("\nText-only mode — skipping audio generation.")
        print(f"Edit {ROLES_CONFIG} to fill in audio_file paths, then re-run with --audio-only.")
        return

    # ── Step 3: register voices ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 3 — Register role voices")
    print("─" * 60)

    roles = load_roles_config()

    # Check API availability
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{INDEXTTS_API_URL}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                if r.status != 200:
                    raise aiohttp.ClientError(f"status {r.status}")
    except Exception as exc:
        print(f"\nERROR: Cannot reach IndexTTS API at {INDEXTTS_API_URL}: {exc}")
        print("Start the API server first:  ./indextts-venv-py311/bin/python indextts-api.py")
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        voice_map = await register_voices(session, roles)

    # ── Step 4: audio generation ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"Step 4 — Generating audio  → {OUTPUT_DIR}/")
    print("─" * 60)

    results = await generate_all_audio(segments, voice_map, OUTPUT_DIR)

    generated = [r for r in results if r]
    skipped   = len(results) - len(generated)

    print("\n" + "─" * 60)
    print(f"Done!  {len(generated)} audio files generated, {skipped} skipped.")
    print(f"Output: {OUTPUT_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ebook-reader demo: text → LLM segmentation → IndexTTS audio",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Run LLM text processing only; do not call the TTS API.",
    )
    parser.add_argument(
        "--audio-only", action="store_true",
        help="Skip LLM; load existing segments.json and generate audio only.",
    )
    args = parser.parse_args()

    if args.text_only and args.audio_only:
        sys.exit("--text-only and --audio-only are mutually exclusive.")

    asyncio.run(main_async(text_only=args.text_only, audio_only=args.audio_only))


if __name__ == "__main__":
    main()
