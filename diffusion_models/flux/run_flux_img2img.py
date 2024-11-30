#%%
import torch
from diffusers import DiffusionPipeline, FluxTransformer2DModel, FluxImg2ImgPipeline
from torchao.quantization import quantize_, int8_weight_only
from diffusers.utils import load_image

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

# load up pipe
pipe = FluxImg2ImgPipeline.from_pretrained(
    model_path
    , transformer = transformer
    , torch_dtype = torch.bfloat16
)
pipe.enable_model_cpu_offload()