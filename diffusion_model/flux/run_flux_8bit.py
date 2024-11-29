#%%
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
import torch

#%%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"

# load transformer model and quantize to int8
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)
quantize_(
    transformer
    , int8_weight_only()
    , device = "cuda:0"
)

#%%
# load up pipe
pipe = DiffusionPipeline.from_pretrained(
    model_path
    , transformer = transformer
    , torch_dtype = torch.bfloat16
)
pipe.enable_model_cpu_offload()

#%%
# generate image
prompt = "4k, best quality, beautiful 20 years girl, show hands and fingers"
image = pipe(
    prompt                  = prompt
    , guidance_scale        = 3.5 
    , num_inference_steps   = 20
    , height                = 1024
    , width                 = 1024
    , generator             = torch.Generator('cuda:0').manual_seed(4)
).images[0]

display(image)