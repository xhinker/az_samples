#%%
import torch
from transformers import AutoModel

# Load omni model (default: init_vision=True, init_audio=True, init_tts=True)
# For vision-only model: set init_audio=False and init_tts=False
# For audio-only model: set init_vision=False
model = AutoModel.from_pretrained(
    "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/llm_models/minicpm-o/models/llm_hf_models/openbmb/MiniCPM-o-4_5_main",
    trust_remote_code=True,
    attn_implementation="sdpa", # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True,
)
model.eval().cuda()

#%% use its tts 
import librosa
model.init_tts()

#%%
ref_audio_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/cosyvoice2/samples/Zh_7_prompt.wav"
ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
sys_msg = {"role": "system", "content": [
  "模仿音频样本的音色并生成新的内容。",
  ref_audio,
  "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"
]}

#%%
# For Chinese
user_msg = {
  "role": "user",
  "content": [
    "请朗读以下内容。" + " " + "你好，欢迎来到艾米说科幻，我是艾米。"
  ]
}

msgs = [sys_msg, user_msg]
res = model.chat(
    msgs=msgs,
    do_sample=True,
    max_new_tokens=512,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.1,
    output_audio_path="result_voice_cloning.wav",
)
print(res)

#%%

#%%
# # Initialize TTS for audio output
# model.init_tts()

# # Convert simplex model to duplex mode
# duplex_model = model.as_duplex()

# # Convert duplex model back to simplex mode
# simplex_model = duplex_model.as_simplex(reset_session=True)