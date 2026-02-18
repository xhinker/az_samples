# Qwen3-TTS Realtime Streaming

This folder now includes a full local streaming demo:
- Python backend: `server.py` (`aiohttp`)
- Web UI: `static/index.html`, `static/styles.css`, `static/app.js`

It supports:
1. Text input
2. Instruction input
3. Reference audio + transcript (voice clone mode)
4. Start/Stop streaming playback in browser

## Install

```sh
pip install -r requirements.txt
```

Optional system dependencies (for audio format support):

```sh
sudo apt-get update && sudo apt-get install sox libsox-fmt-all
```

## Run

From this folder:

```sh
python3 server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --custom-model-id models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice_main \
  --voice-clone-model-id models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main
```

Then open:

```text
http://127.0.0.1:8080
```

## Environment Overrides

- `QWEN_TTS_CUSTOM_MODEL_ID` (or legacy `QWEN_TTS_MODEL_ID`) default: `models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice_main`
- `QWEN_TTS_VOICE_CLONE_MODEL_ID` default: `models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main`
- `QWEN_TTS_DEVICE` (default auto: `cuda:0` if available else `cpu`)
- `QWEN_TTS_DTYPE` (default: `bfloat16`)
- `QWEN_TTS_ATTN_IMPL` (default: `flash_attention_2`)

## Notes

- The backend streams `pcm16le` bytes over HTTP chunked response.
- Browser playback uses Web Audio API scheduling for low-latency continuous audio.
- Stream generation is implemented as incremental text-chunk generation for compatibility with public `qwen_tts` APIs.
