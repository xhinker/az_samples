#%%
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path            = "repos/index-tts/checkpoints/config.yaml"
    , model_dir         = "repos/index-tts/checkpoints"
    , use_fp16          = False
    , use_cuda_kernel   = False
    , use_deepspeed     = False
)

#%%
text = "Translate for me, what is a surprise!"
tts.infer(
    spk_audio_prompt    = 'input_samples/voice_01.wav'
    , text              = text
    , output_path       = "gen.wav"
    , verbose           = True
)