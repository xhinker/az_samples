# Qwen3 ASR Notes

This project is aiming to use Qwen3 ARS to hear sound lively

## TODO
- [x] Run a test using venv
- [ ] Read and study: https://github.com/QwenLM/Qwen3-ASR, then build an Speech to text application in folder: /home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-asr. here are the requirements:
* Build an ASR backend service using python aiohttp, model inference using transformer, not vllm. 
* use model file located in: /home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-asr/models/tts_hf_models/Qwen/Qwen3-ASR-1.7B_main
* The backend service expose speech to text API 
* Build the service can do speech to text streaming, I mean real streaming, that will spit text for every forward inference.
* Build a Web UI using pure html, css and Javascript. in the UI, when I click hear button, the UI will start hearing my current mic device, I am using Chrome in a MacBook


## Notes

* vllm is like a shit, never works with RTX 5090