'''
The 4bit is not working right now
'''

#%%
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int4_weight_only
import torch

#%%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"

# load transformer model and quantize to int8
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)
quantize_(transformer, int4_weight_only())

#%%
# load up pipe
pipe = DiffusionPipeline.from_pretrained(
    model_path
    , transformer = transformer
    , torch_dtype = torch.bfloat16
).to('cuda:0')
# 4bit can't turn on cpu offload
# pipe.enable_model_cpu_offload()

#%%
# generate image
prompt = "4k, best quality, beautiful 20 years girl, show hands and fingers"
image = pipe(
    prompt                  = prompt
    , guidance_scale        = 3.5 
    , num_inference_steps   = 20
    , height                = 1024
    , width                 = 1024
    , generator             = torch.Generator('cuda:0').manual_seed(2)
).images[0]

display(image)