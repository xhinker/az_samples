# index tts readme

- [x] under folder:`/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts` Create an TTS application, so that it will convert text to speech using `index-tts`, following these requirements: 
    * Use model weights under: folder `checkpoints` and `models/tts_hf_models/IndexTeam/IndexTTS-2_main`.
    * Build a real streaming tts service file in file `audio_gen.py`. Real streaming means the generated token and decoded voice data will be emitted during the model forward function. 
    * Build an HTTP TTS API service in file `indextts-api.py` compatible with OpenAI TTS API format, the service should use Python and aiohttp module.
    * Use aiohttp to build Web UI server called: `indextts-server.py`, then create a WebUI using pure HTML, CSS, and Javascript, I can use the WebUI to copy paste text into the input box, upload my reference voice to streaming out the speech. 
    * Use Python venv: `/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts/indextts-venv-py311`
    * API use 8085 port, web server use 8086 port
 
- [ ] 