# Readme

This is a project under current qwen3-tts folder is aiming to build a real time tts that can stream text to voice in real time using Qwen3-tts. User can 
1. Paste custom text and stream to audio
2. Streaming the text output from an LLM and talk to user real time

## TODO
- [ ] Create an `aiohttp` server under `ai_models_eval/voice_models/qwen3-tts` that hosts a web page and streams realtime audio using local model path `models/tts_hf_models/OpenMOSS-Team/Qwen3-TTS-12Hz-1.7B-CustomVoice_main`. reference the Qwen3-tts repo: `/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/moss-tts/repos/MOSS-TTS`
In the web page, user can:
1. provide the text
2. referenced audio
3. audio instruction
4. Start stream, and stop button


## Additional notes
```sh
sudo apt-get update && sudo apt-get install sox libsox-fmt-all
```