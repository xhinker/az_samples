# pip install huggingface_hub

#%%
model_type  = "tts_hf_models"
model_id    = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
branch      = "main"

local_dir   = f"/mnt/data_2t/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-tts/models/{model_type}/{model_id}_{branch}"

cmd_template = rf"""
source /mnt/data_2t/az_git_folder/az_samples/azsample_venv_p312/bin/activate 
huggingface-cli download {model_id} --revision {branch} --local-dir {local_dir}
"""

print(cmd_template)

#%%
model_type  = "tts_hf_models"
model_id    = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
branch      = "main"

local_dir   = f"/mnt/data_2t/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-tts/models/{model_type}/{model_id}_{branch}"

cmd_template = rf"""
source /mnt/data_2t/az_git_folder/az_samples/azsample_venv_p312/bin/activate 
huggingface-cli download {model_id} --revision {branch} --local-dir {local_dir}
"""

print(cmd_template)