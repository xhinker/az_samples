#%%
import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc
from optimization import optimize_pipeline_

# MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
MODEL_ID = "/mnt/data_1t_2/video_models/Wan2.2-I2V-A14B-Diffusers_main"

LANDSCAPE_WIDTH     = 832
LANDSCAPE_HEIGHT    = 480
MAX_SEED            = np.iinfo(np.int32).max

FIXED_FPS           = 16
MIN_FRAMES_MODEL    = 8
MAX_FRAMES_MODEL    = 81

MIN_DURATION        = round(MIN_FRAMES_MODEL/FIXED_FPS,1)
MAX_DURATION        = round(MAX_FRAMES_MODEL/FIXED_FPS,1)

#%%
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID
    , torch_dtype = torch.bfloat16
)

#%%
for i in range(3): 
    gc.collect()
    torch.cuda.synchronize() 
    torch.cuda.empty_cache()

#%%
optimize_pipeline_(
    pipe
    , image           = Image.new('RGB', (LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT))
    , prompt          = 'prompt'
    , height          = LANDSCAPE_HEIGHT
    , width           = LANDSCAPE_WIDTH
    , num_frames      = MAX_FRAMES_MODEL
)

#%%
default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
