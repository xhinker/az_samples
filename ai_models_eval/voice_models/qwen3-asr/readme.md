# Qwen3 ASR Notes

This project is aiming to use Qwen3 ARS to hear sound lively

## TODO
- [x] Run a test using venv
- [x] Build streaming ASR backend + Web UI (see server.py + index.html)
- [ ] Make the following updates:
    1. Support only English and Chinese
    2. WebUI will start sending audio only when there are sound detected, otherwise just stay aware.
    3. Use Audio pause instead of the time chunk size, to make sure a whole sentence is sent for transcription to improve the quality, set 15 seconds as the bar, any audio reach 15 seconds without pause will be sent for transcription.
    4. In the transcription box, show the new result in a new line. 

## Architecture

```
server.py      – aiohttp backend (port 8765)
index.html     – Pure HTML/CSS/JS web UI
```

### Backend endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| POST | `/api/transcribe` | Upload audio → full JSON transcription |
| POST | `/api/transcribe/stream` | Upload audio → SSE token stream |
| GET | `/ws` | WebSocket for real-time mic streaming |

### How streaming works (transformers backend)
1. `TextIteratorStreamer` from HuggingFace intercepts each generated token
2. `model.generate()` runs in a background thread
3. Tokens arrive in an `asyncio.Queue` via `call_soon_threadsafe`
4. The aiohttp WebSocket handler streams each token to the browser immediately

### Run
```bash
source /home/andrewzhu/storage_1t_1/az_git_folder/az_samples/samplecode_venv_p311/bin/activate
python server.py
# then open http://localhost:8765 in Chrome
```


## Notes

* vllm is like a shit, never works with RTX 5090