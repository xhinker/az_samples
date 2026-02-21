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

text = """
这时，一个念头闪过我的脑海，“我看这样吧，咱们附近那个大超市肯定没关门，不如去超市弄点好吃的？”
　　六只眼睛滴溜溜地转了转，好，就这么办！于是三人又冲进了超市，尽管买，尽管拿。水果、熟猪蹄、卤鸭子、烤鸡翅、饭团、炒面、窝窝头……，一人拎了一大口袋。
　　“没地儿咱们就回宿舍吃吧。”黎男提议。
　　ok，三人又蹒跚着回到了宿舍，东西一摊开，整个书桌上下连同床铺，全部被铺满。
　　“师姐，今年的年终奖怎么那么少啊？”师妹边啃着鸭腿边问道，那满嘴儿的油都快顺着下巴滴下去了。
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