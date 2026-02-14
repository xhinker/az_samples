# use py11 venv

#%%
import sys
import os
from diffusers import FluxFillPipeline,FluxTransformer2DModel
import torch
from PIL import Image
import numpy as np
import argparse
import random
from torchao.quantization import (
    quantize_
    , int8_weight_only
)

pipe_gpu_id = 1
quantize_device = f"cuda:{pipe_gpu_id}"

flux_fill_model_path    = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Fill-dev_main"
# no need lora
# redcraft_fill_path      = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_checkpoints/redcraftCADSUpdatedMay11_redEditIcedit11.safetensors"
# need lora, good at NSFW
redcraft_fill_path      = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_checkpoints/redcraftCADSUpdatedJUN1_f1FillNSFW.safetensors"
ice_lora_path           = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/icedit_mormal_lora/pytorch_lora_weights.safetensors"

#%%
transformer = FluxTransformer2DModel.from_single_file(
    redcraft_fill_path
    , subfolder     = "transformer"
    , torch_dtype   = torch.bfloat16
)

pipe = FluxFillPipeline.from_pretrained(
    flux_fill_model_path
    , transformer = transformer
    , torch_dtype = torch.bfloat16
)

pipe.load_lora_weights(
    pretrained_model_name_or_path_or_dict = ice_lora_path
    , adapter_name = "ice_edit"
)

quantize_(
    transformer
    , int8_weight_only() 
    , device = quantize_device # quantize using GPU to accelerate the speed
)

pipe.enable_model_cpu_offload(gpu_id = pipe_gpu_id)

#%%
source_image_path = "/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/output_seed_949.png"
image = Image.open(source_image_path)
image = image.convert("RGB")

set_width = 512
if image.size[0] != set_width:
    print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
    new_width = set_width
    scale = new_width / image.size[0]
    new_height = int(image.size[1] * scale)
    new_height = (new_height // 8) * 8  
    image = image.resize((new_width, new_height))
    print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")
display(image)

width, height   = image.size
combined_image  = Image.new("RGB", (width * 2, height))
combined_image.paste(image, (0, 0))
combined_image.paste(image, (width, 0))
display(combined_image)

mask_array              = np.zeros((height, width * 2), dtype=np.uint8)
mask_array[:, width:]   = 255 
mask = Image.fromarray(mask_array)
display(mask)

#%%
pipe.scheduler

#%%
# instruction     = "change cloth base color to blue"
# instruction     = "change cheongsam color to flowery green"
# instruction     = "Remove her clothes"
instruction     = "show round perky breasts"
# instruction     = "expose breasts"
# instruction     = "show breasts and nipples"
instruction     = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}'

pipe.set_adapters(
    adapter_names       = ['ice_edit']
    , adapter_weights   = [1.0]
)

# from diffusers import FlowMatchEulerDiscreteScheduler
# pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

seed = 17
for i in range(seed, seed+3):
    result_image = pipe(
        prompt                  = instruction
        ,image                  = combined_image
        ,mask_image             = mask
        ,height                 = height
        ,width                  = width * 2
        ,guidance_scale         = 60 #3.5 #3.5#50
        ,num_inference_steps    = 20 #20 #15#28
        ,generator              = torch.Generator("cpu").manual_seed(i)
    ).images[0]
    display(result_image)