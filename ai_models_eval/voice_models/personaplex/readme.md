# PersonaPlex-7B-v1 (NVIDIA)

This folder contains a minimal offline runner for PersonaPlex-7B-v1 using NVIDIA's `moshi` package.

Prereqs
1. Request access to the gated HF model repo and log in (e.g., `huggingface-cli login`).
2. Install `moshi` from the NVIDIA repo:

```sh
git clone https://github.com/NVIDIA/personaplex
cd personaplex/moshi
pip install -e .
```

Optional: list available voice prompts (downloads `voices.tgz`):

```sh
python run_personaplex_7b_v1.py --list-voices
```

Test cases
1. Text input -> voice out (uses a short silence wav as input):

```sh
python run_personaplex_7b_v1.py text_to_voice \
  --text "Hello! Give me a short greeting." \
  --output-wav outputs/text_to_voice.wav
```

2. Voice input -> voice out:

```sh
python run_personaplex_7b_v1.py voice_to_voice \
  --input-wav inputs/user.wav \
  --output-wav outputs/voice_to_voice.wav
```

3. Voice input -> text out:

```sh
python run_personaplex_7b_v1.py voice_to_text \
  --input-wav inputs/user.wav \
  --output-text outputs/voice_to_text.txt
```

Notes
- Input and output audio use `.wav` only.
- `text_to_voice` uses a silence wav as the input channel and injects the user text into the system prompt.
- Default sample rate is 24 kHz; other rates should work, but 24 kHz is recommended.
