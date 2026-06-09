# Higgs Audio v3 TTS — Realtime Streaming Service

OpenAI-compatible TTS API built on **bosonai/higgs-audio-v3-tts-4b** (transformers port). Supports real-time streaming audio, voice cloning, and a web test page.

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| Generation engine | `higgs_audio_gen.py` | AR loop with periodic decode → PCM16 chunks |
| API server | `higgs_tts_api_server.py` | aiohttp, OpenAI-compatible endpoints |
| Web test page | `static/index.html`, `app.js`, `styles.css` | Browser-based streaming playback via Web Audio API |

**Model:** HiggsMultimodalQwen3ForConditionalGeneration (Qwen3-4B backbone + multi-codebook head)  
**Audio codec:** 8 codebooks × 1026 vocab, delay pattern, 24 kHz output  
**Sample rate:** 24 kHz mono PCM16

## Install

```sh
cd /home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/higgs-audio-v3-tts
source higgs_venv/bin/activate
pip install -r requirements.txt
```

Optional (for MP3 conversion):

```sh
sudo apt-get update && sudo apt-get install ffmpeg
```

---

## 1. Start the TTS Service

### Basic start (balanced multi-GPU placement)

```sh
python3 higgs_tts_api_server.py \
  --host 0.0.0.0 \
  --port 8081 \
  --model-path /mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b \
  --device cuda:1 \
  --dtype bfloat16
```

With balanced placement (default), the model auto-distributes across all available GPUs + CPU offload. This handles setups where some GPUs are partially occupied by other processes (e.g., llama-server).

### With explicit GPU memory limit (CPU offload)

Use `--max-gpu-memory` to force CPU offloading on low-VRAM setups:

```sh
python3 higgs_tts_api_server.py \
  --host 0.0.0.0 \
  --port 8081 \
  --model-path /mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b \
  --device cuda:1 \
  --dtype bfloat16 \
  --max-gpu-memory 4GiB
```

### Environment variable overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `HIGGS_MODEL_PATH` | `/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b` | Model directory path |
| `HIGGS_DEVICE` | `cuda:1` | Target GPU device |
| `HIGGS_DTYPE` | `bfloat16` | Model precision (`bfloat16` or `float16`) |

Example with env vars:

```sh
export HIGGS_MODEL_PATH=/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b
export HIGGS_DEVICE=cuda:1
python3 higgs_tts_api_server.py --port 8081
```

### Verify the service is running

```sh
curl http://localhost:8081/v1/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/mnt/data_2t_3/ai_models_all/higgs-audio-v3-tts-4b",
  "sample_rate": 24000,
  "device": "cuda:1"
}
```

---

## 2. Test with CLI curl

### Basic TTS (WAV streaming → save file)

```sh
curl -X POST http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of the Higgs Audio TTS system.",
    "voice": "alloy",
    "response_format": "wav",
    "temperature": 0.8
  }' \
  --output output.wav
```

### PCM raw stream (no header)

```sh
curl -X POST http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world.",
    "voice": "echo",
    "response_format": "pcm",
    "temperature": 0.7
  }' \
  --output output.raw
# Play with: aplay -r 24000 -f S16_LE -t raw output.raw
```

### MP3 (non-streaming, full conversion)

```sh
curl -X POST http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "voice": "nova",
    "response_format": "mp3",
    "temperature": 0.85
  }' \
  --output output.mp3
```

### Voice cloning with reference audio

Encode the reference WAV as base64 and include it in the request:

```sh
REF_B64=$(base64 -w0 /path/to/reference.wav)

curl -X POST http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"Have a nice day and enjoy south california sunshine.\",
    \"voice\": \"alloy\",
    \"response_format\": \"wav\",
    \"reference_audio\": \"$REF_B64\",
    \"reference_text\": \"Hey, Adam here. Lets create something that feels real.\"
  }" \
  --output cloned_output.wav
```

### Emotion & style control (inline tags)

```sh
curl -X POST http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. <|sfx:laughter|>Hehe, no seriously.",
    "voice": "alloy",
    "response_format": "wav",
    "temperature": 0.8
  }' \
  --output emotion_output.wav
```

### List available models

```sh
curl http://localhost:8081/v1/models
```

---

## 3. Test with Python Script

Save as `test_tts.py` and run from the project directory.

### Basic streaming test

```python
#!/usr/bin/env python3
"""Basic TTS streaming test — saves PCM chunks to WAV file."""
import requests
import wave
import struct

URL = "http://localhost:8081/v1/audio/speech"

payload = {
    "input": "Hello, this is a quick test of the Higgs Audio TTS system.",
    "voice": "alloy",
    "response_format": "pcm",
    "temperature": 0.8,
}

with requests.post(URL, json=payload, stream=True) as resp:
    assert resp.status_code == 200, f"Error: {resp.text}"

    all_pcm = b""
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            all_pcm += chunk
            # Real-time processing here (e.g., play via sounddevice)

# Save as WAV
sr = int(resp.headers.get("X-Sample-Rate", 24000))
with wave.open("test_output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(all_pcm)

duration = len(all_pcm) / (sr * 2)
print(f"Generated {len(all_pcm)} bytes (~{duration:.1f}s audio)")
```

### Voice cloning test

```python
#!/usr/bin/env python3
"""Voice cloning test with reference audio."""
import requests
import base64
import wave

URL = "http://localhost:8081/v1/audio/speech"

# Load reference audio as base64
with open("reference.wav", "rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode()

payload = {
    "input": "Have a nice day and enjoy south california sunshine.",
    "voice": "alloy",
    "response_format": "pcm",
    "reference_audio": ref_audio_b64,
    "reference_text": "Hey, Adam here. Lets create something that feels real.",
    "temperature": 0.8,
}

with requests.post(URL, json=payload, stream=True) as resp:
    assert resp.status_code == 200

    all_pcm = b""
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            all_pcm += chunk

sr = int(resp.headers.get("X-Sample-Rate", 24000))
with wave.open("cloned_output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(all_pcm)

print(f"Cloned audio: {len(all_pcm)} bytes (~{len(all_pcm)/(sr*2):.1f}s)")
```

### Real-time playback test (requires `sounddevice`)

```python
#!/usr/bin/env python3
"""Real-time TTS streaming with live playback."""
import requests
import sounddevice as sd

URL = "http://localhost:8081/v1/audio/speech"
SR = 24000

payload = {
    "input": "Hello, this is a real-time streaming test.",
    "voice": "alloy",
    "response_format": "pcm",
    "temperature": 0.8,
}

with requests.post(URL, json=payload, stream=True) as resp:
    assert resp.status_code == 200

    stream = sd.OutputStream(samplerate=SR, channels=1, dtype="int16")
    stream.start()

    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            stream.write(chunk)

    stream.stop()
    stream.close()

print("Playback complete.")
```

Install sounddevice: `pip install sounddevice`

---

## 4. Test with Web Page

The built-in web page provides a full interactive TTS testing interface with real-time browser playback via Web Audio API.

### Start the server (if not already running)

```sh
python3 higgs_tts_api_server.py --host 0.0.0.0 --port 8081
```

### Open in browser

Navigate to: **http://localhost:8081** (or http://\<server-ip\>:8081 for remote access)

### Web page features

| Feature | Details |
|---------|---------|
| **Text input** | Multi-line textarea for any text (supports 100+ languages including Chinese) |
| **Voice selection** | Alloy / Echo / Fable / Onyx / Nova / Shimmer — each maps to different temperature presets |
| **Temperature slider** | Range 0.1–2.0 with live value display (default 0.8) |
| **Output format** | WAV (streaming), PCM (raw streaming), MP3 (non-streaming download) |
| **Streaming mode** | Chunked Web Audio API playback for sub-second latency |
| **Voice cloning** | Upload reference `.wav`/`.mp3` file + optional transcript text |
| **Start / Stop buttons** | Start generates and plays audio; Stop cancels both server-side generation and client playback |
| **Status log** | Timestamped events showing stream progress and byte counts |

### Testing workflow

1. Enter text in the input field (or paste Chinese/Japanese/etc.)
2. Select voice and adjust temperature if desired
3. Click **▶ Generate** — audio starts playing within ~1 second
4. Audio streams continuously as it's generated on the server
5. Click **⏹ Stop** at any time to cancel both generation and playback
6. For voice cloning: expand the "Voice Cloning (optional)" section, upload a reference audio file, optionally add the transcript text, then generate

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Health check — model status, device info |
| `GET` | `/v1/models` | List available models (OpenAI-compatible) |
| `POST` | `/v1/audio/speech` | TTS synthesis with optional streaming |

### POST /v1/audio/speech request body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | **Yes** | — | Text to synthesize (supports inline control tokens) |
| `model` | string | No | any | Model ID (ignored — always uses higgs-audio-v3-tts) |
| `voice` | string | No | `alloy` | Voice preset: alloy/echo/fable/onyx/nova/shimmer |
| `response_format` | string | No | `wav` | Output format: pcm/wav/mp3 |
| `temperature` | float | No | per voice | Sampling temperature (0.1–2.0) |
| `top_p` | float | No | None | Nucleus sampling threshold |
| `top_k` | int | No | None | Top-k sampling limit |
| `reference_audio` | string | No | — | Base64-encoded WAV bytes for voice cloning |
| `reference_text` | string | No | — | Transcript of reference audio (improves clone quality) |

### Inline control tokens

Embed these in the `input` text:

- **Emotion:** `<|emotion:elation|>`, `<|emotion:sadness|>`, `<|emotion:anger|>`, etc.
- **Style:** `<|style:singing|>`, `<|style:whispering|>`
- **Prosody:** `<|prosody:speed_fast|>`, `<|prosody:pitch_high|>`, `<|prosody:pause|>`
- **Sound effects:** `<|sfx:laughter|>Haha` (always pair with onomatopoeia)

---

## Notes

- The backend streams raw PCM16LE bytes over HTTP chunked transfer for WAV and PCM formats — no WAV header is sent during streaming to avoid browser playback issues.
- Browser playback uses Web Audio API scheduling for low-latency continuous audio (< 1 second time-to-first-audio).
- Balanced multi-GPU device placement distributes the model across all available GPUs + CPU offload by default. Use `--max-gpu-memory` for explicit memory limits.
- One active stream at a time (async lock prevents concurrent generation conflicts).
- Stop button cancels server-side generation via threading.Event — not just client playback.
