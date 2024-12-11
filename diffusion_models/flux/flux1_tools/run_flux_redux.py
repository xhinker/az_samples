'''
## What is this
Sample code to use Flux Redux, It is similar to the IPAdapter

## Prepare

Update diffusers to the newest version: `pip install -U diffusers`

## Changes
Created on: Dec 1st, 2024
'''

#%%
import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
from diffusers.utils import load_image
from azailib.sd_pipe_loaders import (
    load_flux1_img2img_8bit_pipe
    , load_flux1_8bit_pipe
)
from azailib.image_tools import resize_img

base_model_path         = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
transformer_model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_checkpoints/fluxescoreDev_asian_v10Fp16"
redux_model_path        = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Redux-dev_main"
device                  = "cuda:0"

#%% load pipe using azailib
pipe = load_flux1_img2img_8bit_pipe(
    checkpoint_path_or_id = base_model_path
    , pipe_gpu_id = 0
)

#%% load redux prior
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    redux_model_path
    , torch_dtype = torch.bfloat16
).to(device)

#%% load image and resize
image_path  = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/model.png"
init_image  = load_image(image_path)
# resize image
input_img   = resize_img(img_path=image_path, upscale_times=0.75)
width, height = input_img.size
print(width, height)
input_img

#%%
image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/dress.png"
image               = load_image(image_path)
display(image)

pipe_prior_output   = pipe_prior_redux(image)

#%%
image = pipe(
    image                   = input_img
    , width                 = width
    , height                = height
    , guidance_scale        = 4.5
    , num_inference_steps   = 20
    , strength              = 0.8
    , generator             = torch.Generator("cpu").manual_seed(0)
    , **pipe_prior_output
).images[0]
image