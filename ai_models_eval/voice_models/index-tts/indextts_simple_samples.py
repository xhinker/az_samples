#%%
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path            = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts/repos/index-tts/checkpoints/config.yaml"
    , model_dir         = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts/repos/index-tts/checkpoints"
    , use_fp16          = True #False
    , use_cuda_kernel   = False #False
    , use_deepspeed     = False
)

#%% 
text = """
　　“师姐，那我们怎么办呢？”师妹噘着小嘴儿。
　　“我看今天在这附近很难找到饭馆了。”我四下里望了望，那一排排的门面，一个个大门紧闭。
　　“我看我们打个车进市区吧，市区应该有吃饭的地儿。”黎男建议道。
"""

tts.infer(
    spk_audio_prompt    = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts/input_samples/Zh_7_prompt.wav'
    , text              = text
    , output_path       = "gen_2.wav"
    , verbose           = True
)

#%% emotion control
text = """
　　“师姐，那我们怎么办呢？”师妹噘着小嘴儿。
　　“我看今天在这附近很难找到饭馆了。”我四下里望了望，那一排排的门面，一个个大门紧闭。
"""
emo_text = "非常难过, 非常伤心"
tts.infer(
    spk_audio_prompt    = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/index-tts/input_samples/Zh_7_prompt.wav'
    , text              = text
    , output_path       = "gen_emo_1.wav"
    , verbose           = True
    , emo_alpha         = 0.6
    , use_emo_text      = True
    , emo_text          = emo_text
    , use_random        = False
)