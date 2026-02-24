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
	上星期的某一天晚上，我和一位朋友在西单附近吃饭。席间我们高谈阔论，指点江山，臧否人物，言必及王小波、余杰、村上春树、奥尔罕·帕慕克，聊的十分尽兴。大约到了9点多，我们方才起身结帐，各自回家。我踏上地铁之前，忽然看到一处还没收摊的报刊亭，就走了过去。从西单到四惠东大约11站，全程要30多分钟，我必须得买点什么东西消遣。

	我的视线从《科学美国人》扫到《译林》，然后又从《看电影》扫到《三联文化周刊》，来回溜达了五、六分钟仍旧游移不决，直到摊主不耐烦说要收摊了，我才催促自己下了决心，在摊子上抓了一本《读者》，匆匆离去。在地铁里，我捧着《读者》看的津津有味，全然不顾自己曾经一逮着机会就嘲讽这本杂志的种种劣行。《读者》杀时间很是不错，我在西单等地铁的时候翻开扉页寄语，在建国门看到中缝后的笑话栏目，然后四惠东地铁停稳的一瞬间，我刚好扫完封底的广告。

	尽管我一下车就把《读者》顺手塞进垃圾筒内，扬长而去，但我必须得承认：我在刚才的30分钟过的很愉悦，那些小布尔乔亚式温情故事和心灵鸡汤让我发酵出一种中产阶级的微微醺意。

	我上上星期去了一趟三联书店，用公司发的雅高卡买了许多一直想要但很贵的书，比如王鸣盛的《十七史商榷》、张岩的《审核古文案》、杨宽的《中国古代都城制度史》、《百变小红帽－一则童话三百年的演变》，还有若干本“大家小书”系列的小册子。买新书是一件令人愉悦的事，尤其是买了这么多看起来既深沉又有内涵的文化书籍之后，感觉旁人注视自己的眼神都多了几分恭敬。我捧着这些书兴致勃勃地回到家里，把它们一本一本摆在书架上，心里盘算哪些书以后写东西用得着；哪些书以后吹牛用得着；哪些书可以增加自己的修为和学问。

	盘算到一半的时候，腹中忽有触动，五谷轮回，山雨欲来。我的视线飞过这些崭新的内涵书，抽出一本机器猫，匆忙跑进厕所……

	类似的事情其实经常发生。比如跑去看现代艺术画展，最后发现真正停留超过两分钟欣赏的，都是裸女主题油画；买来许多经典dvd，最后挑拣出来搁进影碟机的只有《恐怖星球》和《料理鼠王》，看到男主角居然是大厨古斯特的私生子时，还乱感动了一把；往psp里灌了300多种历代典籍文献，然后只是一味玩《分裂细胞》——甚至当我前天偶尔在手机里下载了一款类似口袋妖怪的java游戏以后，我连psp都不玩了，每天在班车上和地铁里不停地按动手机键，就如同一位真正的无聊上班族。

	我有一次看到《little britannia》里有个桥段：男主角之一跑去一家高级法国餐厅吃饭，对着白发苍苍的老侍应生说：“给我来份加大的麦辣汉堡。”这让我亲切莫名。

	我把这个发现跟朋友们说，他们都纷纷表示自己也有类似的经历。有人拟定了全套瑜珈健身计划，然后周末在家里睡足两天；还有人买了精致的手动咖啡磨，然后摆在最醒目的位置，继续喝速溶伴侣。最后大家一起唉声叹气，试图要把这个发现上升到哲学高度，提炼出一点什么精神感悟，让自己上个层次什么的。

	但是这个努力可耻地失败了，于是我们发现这是一种感染范围很广泛的疾病。

	简单来说，下里巴症候群是这样一种病：我们会努力要作一个风雅的人、一个高尚的人，一个脱离了低级趣味的人，结果还是在最不经意的时候暴露出自己的俗人本质。我们试图跟着阳春白雪的调子高唱，脑子里想的却总是阳春面和白雪公主。

	一般这种疾病分成两个阶段：第一个阶段是你发现了“超我”，折射到现实社会，就是你买了一台西电ks-16608l；第二个阶段是你发现了“本我”，每天晚上都用这玩意儿听《两只蝴蝶》。

	其实仔细想想，这种疾病或者说生活状态很不错，一来可以满足自己的虚荣心；二来又不会真正让自己难受——要知道，让一个俗人去勉强风雅，比让一个风雅的人勉强去俗气更不容易，毕竟不是每个人都象郭沫若那样进退自如，能写出《凤凰涅磐》和《咒麻雀》来。

	按照文法，在文章的结尾应该提纲挈领，但是刚才已经失败了，现在也不会有什么成功的可能。所以我还是以一个隽永温馨的哲理小故事作为结尾。

	我有一个朋友r。有一次，我们一群人去看一部话剧。当时去的早了，话剧还没开演。百无聊赖之下，我们就跑到附近的一家书店闲逛。我偶尔瞥到其中一个书架上放着一些关于佛教的书，忽然下里巴症又发作，于是微皱眉头，用轻松安详的语气说恰好在旁边的a说：“最近俗务缠身，我忽然很想看看禅宗的精神，让自己的心空一下，也未尝不是件愉悦的事。”

	y没理我。我低头一看，r原来正蹲在地上，聚精会神地捧着从书架角落里拿出来的大书。

	“你在看什么？”

	a把书举了起来，我首先看到的是y愉悦的表情，然后是封面硕大的字体：“慈禧美容秘籍。”

	r的真诚和坦率就如同初春的阳光，我看到自己虚伪的面具惭愧地开始融化。心灵被震撼的我扔下了南怀瑾、南怀仁和慧能，毫无矫饰地抽出一本《奇侠杨小邪》。

	我的内心学着《发条橙》结尾的阿历克斯，大声呐喊：“i was cured all right。”
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
