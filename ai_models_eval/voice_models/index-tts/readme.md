# index tts readme

- [x] under folder:`/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts` Create an TTS application, so that it will convert text to speech using `index-tts`, following these requirements: 
    * Use model weights under: folder `checkpoints` and `models/tts_hf_models/IndexTeam/IndexTTS-2_main`.
    * Build a real streaming tts service file in file `audio_gen.py`. Real streaming means the generated token and decoded voice data will be emitted during the model forward function. 
    * Build an HTTP TTS API service in file `indextts-api.py` compatible with OpenAI TTS API format, the service should use Python and aiohttp module.
    * Use aiohttp to build Web UI server called: `indextts-server.py`, then create a WebUI using pure HTML, CSS, and Javascript, I can use the WebUI to copy paste text into the input box, upload my reference voice to streaming out the speech. 
    * Use Python venv: `/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts/indextts-venv-py311`
    * API use 8085 port, web server use 8086 port
 
- [x] I want to build a ebook reader with role play sounds, so that different role will use different role sound as reference audio for indextts 
* now build a text processor `text_processor.py` using LLM to convert text to a list of json object like this: 
```json
[
    {
        "role":"role_name",
        "gender":"male/female",
        "emotion:":"the emotion description",
        "type":"dialog/narrator/environment_sound",
        "text":"text to convert to speech"
    },
    ...
]
```
* Use the following LLM:
```
DEFAULT_LLM_API_URL = "http://192.168.68.79:1234/v1"
DEFAULT_LLM_API_KEY = "lmstudio"
DEFAULT_LLM_MODEL_NAME = "qwen/qwen3-coder-next"
```
* Generate a role audio configuration file in this format:
```json
[
    {
        "role":"role_name",
        "gender":"male/female",
        "audio_file":"path_to_the_audio_file"
    }
]
```
I will fill in the audio file later.
* Then create a sample code to test the text convert and use the json to generate audios using the list of generated audio text using indextts. 