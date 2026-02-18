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
- Stream generation now uses internal code-token streaming hooks (CloudWells-style) instead of text chunking.
- For stability, one active stream per mode is processed at a time (`custom_voice` and `voice_clone` each have their own lock).
- `Stop` sends `/api/stop` to cancel server-side generation (not only client playback).
- Long text is automatically re-anchored into ~1-minute segments to avoid quality drift in very long autoregressive runs.

## TODO
- [ ] Leverage audio_gen.py to live stream the output from a LLM model. here is the implementation details: 
1. Create another web page app using pure HTML, CSS and Javascript for LLM chat
2. Reuse the web server called llm_audio_server.py as the backend host, using aiohttp, use port 8085. 
3. The LLM API detail:
    ```json
    {
        "api_name":"macm4max_lmstudio",
        "api_url":"http://192.168.68.79:1234/v1",
        "api_key":"lmstudio",
        "model_name":"qwen/qwen3-coder-next"
    }
    ```
4. In the UI, provide a predefined system prompt to tell LLM to output text that will be read out, and make the system prompt editable so that user can provide customized prompt.
5. In the UI, while text streaming out, also stream the live audio