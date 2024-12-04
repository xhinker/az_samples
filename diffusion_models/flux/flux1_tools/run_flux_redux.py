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

base_model_path         = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
transformer_model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_checkpoints/fluxescoreDev_asian_v10Fp16"
redux_model_path        = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Redux-dev_main"
device                  = "cuda:0"

transformer = FluxTransformer2DModel.from_pretrained(
    transformer_model_path
    , subfolder     = "transformer"
    , torch_dtype   = torch.bfloat16
)
quantize_(
    transformer
    , int8_weight_only() 
    , device = device # quantize using GPU to accelerate the speed
)

#%%
pipe = FluxPipeline.from_pretrained(
    base_model_path
    , transformer   = transformer
    , torch_dtype   = torch.bfloat16
)
pipe.enable_model_cpu_offload()

#%% load redux prior
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    redux_model_path
    , torch_dtype = torch.bfloat16
).to(device)

#%%
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png"
image               = load_image(image_path)
display(image)

pipe_prior_output   = pipe_prior_redux(image)

#%%
image = pipe(
    guidance_scale          = 2.5
    , num_inference_steps   = 30
    , generator             = torch.Generator("cpu").manual_seed(0)
    , **pipe_prior_output
).images[0]
image