# Moss TTS Sample

This is a project aiming to build a real time tts that can stream text to voice in real time. User can 
1. Paste custom text and stream to audio
2. Streaming the text output from an LLM and talk to user real time

## TODO

- [x] Create an `aiohttp` server under `ai_models_eval/voice_models/moss-tts` that hosts a web page and streams realtime audio using local model path `models/tts_hf_models/OpenMOSS-Team/MOSS-TTS-Realtime_main`.
- [x] Fix UI/server import failure: `cannot import name 'initialization' from transformers` by adding compatibility fallback and clear runtime compatibility checks.
- [x] Fix runtime streaming failure: Torch Dynamo `Data-dependent branching` by removing compile path in realtime streaming helper and using a safer attention backend (`sdpa`) in server flow.
- [x] Save a server-side WAV copy when each stream finishes, under peer `outputs/` folder, and report saved path back to UI.
- [x] Split inline UI code from Python server into standalone `realtime_page.html` and serve it from `/`.
- [x] Fix double-click issue (first click only shows connected message, second click starts stream) by resolving WebSocket lifecycle race in client JS.
- [x] Fix streaming playback flicker/gap noise in browser by updating chunk scheduling logic (contiguous scheduling, remove per-chunk fade gating that introduced boundary artifacts).
