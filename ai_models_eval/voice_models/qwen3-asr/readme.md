# Qwen3 ASR Notes

This project is aiming to use Qwen3 ARS to hear sound lively

## TODO
- [x] Run a test using venv
- [x] Build streaming ASR backend + Web UI (see server.py + index.html)

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