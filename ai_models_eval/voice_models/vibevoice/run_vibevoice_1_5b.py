"""Minimal VibeVoice-1.5B inference example.

Usage:
  python3 run_vibevoice_1_5b.py \\
    --text "Hello from VibeVoice" \\
    --output outputs/vibevoice_demo.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import VibeVoiceForConditionalGeneration, VibeVoiceProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TTS with microsoft/VibeVoice-1.5B")
    parser.add_argument(
        "--model-id",
        default="microsoft/VibeVoice-1.5B",
        help="Hugging Face model id",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--speaker",
        default="speaker1",
        choices=["speaker1", "speaker2"],
        help="Speaker role used by the chat template",
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        help="Optional reference audio path for voice conditioning",
    )
    parser.add_argument("--output", default="outputs/vibevoice_output.wav", help="Output wav path")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max generated tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading processor/model: {args.model_id}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_id)
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)

    messages = [{"role": args.speaker, "content": args.text}]
    chat_kwargs = {
        "conversation": messages,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if args.reference_audio:
        chat_kwargs["audio_path"] = [args.reference_audio]

    inputs = processor.apply_chat_template(**chat_kwargs)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    print("Generating audio...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    audio_values = processor.batch_decode(output_ids.cpu(), output_type="pt")[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = getattr(processor, "sampling_rate", 24000)
    torchaudio.save(str(output_path), audio_values.unsqueeze(0), sample_rate)

    print(f"Saved audio to: {output_path}")
    print(f"Sample rate: {sample_rate}")


if __name__ == "__main__":
    main()
