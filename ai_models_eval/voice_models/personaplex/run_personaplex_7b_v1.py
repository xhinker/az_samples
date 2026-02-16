"""Minimal PersonaPlex-7B-v1 offline runner.

Prereqs (from NVIDIA PersonaPlex repo):
1) Clone the repo and install moshi:
   git clone https://github.com/NVIDIA/personaplex
   cd personaplex/moshi
   pip install -e .
2) Make sure you have access to the gated HF model repo and are logged in.

Examples (test cases):
1) Text input -> voice out (uses a short silence wav as input)
   python run_personaplex_7b_v1.py text_to_voice \
     --text "Hello! Give me a short greeting." \
     --output-wav outputs/text_to_voice.wav

2) Voice input -> voice out
   python run_personaplex_7b_v1.py voice_to_voice \
     --input-wav inputs/user.wav \
     --output-wav outputs/voice_to_voice.wav

3) Voice input -> text out
   python run_personaplex_7b_v1.py voice_to_text \
     --input-wav inputs/user.wav \
     --output-text outputs/voice_to_text.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def _import_moshi_offline():
    try:
        from moshi import offline as moshi_offline  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time failure
        raise SystemExit(
            "Could not import 'moshi'. Install it from NVIDIA/personaplex first. "
            f"Original error: {exc}"
        )
    return moshi_offline


def ensure_silence_wav(path: Path, seconds: float = 6.0, sample_rate: int = 24000) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)


def resolve_voice_prompt(moshi_offline, voice_prompt: str | None, voice_prompt_dir: str | None, hf_repo: str) -> Path:
    voice_dir = moshi_offline._get_voice_prompt_dir(voice_prompt_dir, hf_repo)
    if voice_dir is None:
        raise SystemExit("Could not resolve voice prompt directory. Provide --voice-prompt-dir.")
    voice_dir_path = Path(voice_dir)

    if voice_prompt:
        voice_path = Path(voice_prompt)
        if voice_path.is_file():
            return voice_path
        candidate = voice_dir_path / voice_prompt
        if candidate.is_file():
            return candidate
        available = sorted(p.name for p in voice_dir_path.glob("*.pt"))
        raise SystemExit(
            "Voice prompt not found. "
            f"Tried '{voice_prompt}'. Available: {available}"
        )

    candidates = sorted(voice_dir_path.glob("*.pt"))
    if not candidates:
        raise SystemExit(f"No .pt voice prompts found in {voice_dir_path}")
    return candidates[0]


def list_voice_prompts(moshi_offline, voice_prompt_dir: str | None, hf_repo: str) -> None:
    voice_dir = moshi_offline._get_voice_prompt_dir(voice_prompt_dir, hf_repo)
    if voice_dir is None:
        raise SystemExit("Could not resolve voice prompt directory. Provide --voice-prompt-dir.")
    voice_dir_path = Path(voice_dir)
    prompts = sorted(p.name for p in voice_dir_path.glob("*.pt"))
    if not prompts:
        print(f"No voice prompts found in {voice_dir_path}")
        return
    print("Available voice prompts:")
    for name in prompts:
        print(f"- {name}")


def build_text_prompt(persona: str, user_text: str | None) -> str:
    persona = (persona or "").strip()
    if not user_text:
        return persona
    user_text = user_text.strip()
    if persona:
        return f"{persona}\nUser: {user_text}\nAssistant:"
    return f"User: {user_text}\nAssistant:"


def tokens_to_text(tokens: list[str]) -> str:
    special = {"EPAD", "BOS", "EOS", "PAD"}
    pieces = [t for t in tokens if t not in special]
    text = "".join(pieces)
    text = text.replace("\u2581", " ")
    text = " ".join(text.split())
    return text.strip()


def run_inference(
    moshi_offline,
    *,
    input_wav: Path,
    output_wav: Path,
    output_text_json: Path,
    text_prompt: str,
    voice_prompt_path: Path,
    hf_repo: str,
    device: str,
    cpu_offload: bool,
    seed: int,
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    output_text_json.parent.mkdir(parents=True, exist_ok=True)

    moshi_offline.run_inference(
        input_wav=str(input_wav),
        output_wav=str(output_wav),
        output_text=str(output_text_json),
        text_prompt=text_prompt,
        voice_prompt_path=str(voice_prompt_path),
        tokenizer_path=None,
        moshi_weight=None,
        mimi_weight=None,
        hf_repo=hf_repo,
        device=device,
        seed=seed,
        temp_audio=temp_audio,
        temp_text=temp_text,
        topk_audio=topk_audio,
        topk_text=topk_text,
        greedy=greedy,
        save_voice_prompt_embeddings=False,
        cpu_offload=cpu_offload,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NVIDIA PersonaPlex-7B-v1 (offline)")
    parser.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1", help="HF repo id")
    parser.add_argument(
        "--persona",
        default="You are a helpful assistant who speaks clearly and concisely.",
        help="System prompt / persona text",
    )
    parser.add_argument("--voice-prompt", default=None, help="Voice prompt .pt file name or full path")
    parser.add_argument(
        "--voice-prompt-dir",
        default=None,
        help="Directory containing voice prompt .pt files (optional)",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--cpu-offload", action="store_true", help="CPU offload (requires accelerate)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="Audio temperature")
    parser.add_argument("--temp-text", type=float, default=0.7, help="Text temperature")
    parser.add_argument("--topk-audio", type=int, default=250, help="Audio top-k")
    parser.add_argument("--topk-text", type=int, default=25, help="Text top-k")
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")

    subparsers = parser.add_subparsers(dest="case")

    p_text = subparsers.add_parser("text_to_voice", help="Text input -> voice output")
    p_text.add_argument("--text", required=True, help="User text input")
    p_text.add_argument("--output-wav", default="outputs/text_to_voice.wav", help="Output wav path")
    p_text.add_argument("--output-text-json", default=None, help="Optional output json path")
    p_text.add_argument(
        "--silence-wav",
        default="inputs/silence_6s.wav",
        help="Silence wav used as input (auto-created if missing)",
    )
    p_text.add_argument("--silence-seconds", type=float, default=6.0, help="Silence duration")

    p_v2v = subparsers.add_parser("voice_to_voice", help="Voice input -> voice output")
    p_v2v.add_argument("--input-wav", required=True, help="Input wav path")
    p_v2v.add_argument("--output-wav", default="outputs/voice_to_voice.wav", help="Output wav path")
    p_v2v.add_argument("--output-text-json", default=None, help="Optional output json path")

    p_v2t = subparsers.add_parser("voice_to_text", help="Voice input -> text output")
    p_v2t.add_argument("--input-wav", required=True, help="Input wav path")
    p_v2t.add_argument("--output-wav", default="outputs/voice_to_text.wav", help="Output wav path")
    p_v2t.add_argument("--output-text", default="outputs/voice_to_text.txt", help="Output text path")
    p_v2t.add_argument("--output-text-json", default=None, help="Optional output json path")

    args = parser.parse_args()

    if args.list_voices:
        return args
    if args.case is None:
        parser.error("Choose a case: text_to_voice, voice_to_voice, or voice_to_text")
    return args


def main() -> None:
    args = parse_args()
    moshi_offline = _import_moshi_offline()

    if args.list_voices:
        list_voice_prompts(moshi_offline, args.voice_prompt_dir, args.hf_repo)
        return

    voice_prompt_path = resolve_voice_prompt(
        moshi_offline,
        voice_prompt=args.voice_prompt,
        voice_prompt_dir=args.voice_prompt_dir,
        hf_repo=args.hf_repo,
    )

    if args.case == "text_to_voice":
        silence_wav = Path(args.silence_wav)
        ensure_silence_wav(silence_wav, seconds=args.silence_seconds)
        output_wav = Path(args.output_wav)
        output_text_json = Path(args.output_text_json or output_wav.with_suffix(".json"))
        text_prompt = build_text_prompt(args.persona, args.text)

        run_inference(
            moshi_offline,
            input_wav=silence_wav,
            output_wav=output_wav,
            output_text_json=output_text_json,
            text_prompt=text_prompt,
            voice_prompt_path=voice_prompt_path,
            hf_repo=args.hf_repo,
            device=args.device,
            cpu_offload=args.cpu_offload,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=args.greedy,
        )
        print(f"Saved audio to {output_wav}")
        print(f"Saved text tokens to {output_text_json}")
        return

    if args.case == "voice_to_voice":
        input_wav = Path(args.input_wav)
        output_wav = Path(args.output_wav)
        output_text_json = Path(args.output_text_json or output_wav.with_suffix(".json"))
        text_prompt = build_text_prompt(args.persona, None)

        run_inference(
            moshi_offline,
            input_wav=input_wav,
            output_wav=output_wav,
            output_text_json=output_text_json,
            text_prompt=text_prompt,
            voice_prompt_path=voice_prompt_path,
            hf_repo=args.hf_repo,
            device=args.device,
            cpu_offload=args.cpu_offload,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=args.greedy,
        )
        print(f"Saved audio to {output_wav}")
        print(f"Saved text tokens to {output_text_json}")
        return

    if args.case == "voice_to_text":
        input_wav = Path(args.input_wav)
        output_wav = Path(args.output_wav)
        output_text = Path(args.output_text)
        output_text_json = Path(args.output_text_json or output_text.with_suffix(".json"))
        text_prompt = build_text_prompt(args.persona, None)

        run_inference(
            moshi_offline,
            input_wav=input_wav,
            output_wav=output_wav,
            output_text_json=output_text_json,
            text_prompt=text_prompt,
            voice_prompt_path=voice_prompt_path,
            hf_repo=args.hf_repo,
            device=args.device,
            cpu_offload=args.cpu_offload,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=args.greedy,
        )

        try:
            tokens = json.loads(output_text_json.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - IO parsing
            raise SystemExit(f"Failed to read {output_text_json}: {exc}")
        text = tokens_to_text(tokens)
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text(text, encoding="utf-8")

        print(f"Saved audio to {output_wav}")
        print(f"Saved text to {output_text}")
        print(f"Saved text tokens to {output_text_json}")
        return

    raise SystemExit(f"Unknown case: {args.case}")


if __name__ == "__main__":
    main()
