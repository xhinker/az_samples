#%%
#%%
import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from torchao.quantization import quantize_, int8_weight_only

dtype = torch.bfloat16
device = "cuda:0"

# model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
# model_id = "/home/andrewzhu/storage_14t_5/ai_models_all/llm_hf_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers_main"
model_id = "/mnt/data_1t_2/video_models/Wan2.2-TI2V-5B-Diffusers_main"
vae = AutoencoderKLWan.from_pretrained(
    model_id
    , subfolder                 = "vae"
    , torch_dtype               = torch.float32
    , low_cpu_mem_usage         = False
    , ignore_mismatched_sizes   = True
)

#%%
pipe = WanPipeline.from_pretrained(
    model_id
    , vae           = vae
    , torch_dtype   = dtype
    # , low_cpu_mem_usage=False    
    # , ignore_mismatched_sizes=True
)

#%%
transformer     = pipe.transformer
transformer_2   = pipe.transformer_2

quantize_(
    transformer
    , int8_weight_only()
    , device = "cuda:0"
)

quantize_(
    transformer_2
    , int8_weight_only()
    # , int4_weight_only()
    , device = "cuda:0"
)

# pipe.to(device)
pipe.enable_model_cpu_offload()

#%%
height = 704
width = 1280
num_frames = 121
num_inference_steps = 50
guidance_scale = 5.0

#%%

prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
).frames[0]
export_to_video(output, "5bit2v_output.mp4", fps=24)