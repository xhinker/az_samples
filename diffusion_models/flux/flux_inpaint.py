'''
Need verify, TBD
hmm, what is this.
'''

#%%
import torch
from diffusers import FluxControlNetInpaintPipeline,FluxTransformer2DModel
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image
from torchao.quantization import (
    quantize_
    , int8_weight_only
)

controlnet = FluxControlNetModel.from_pretrained(
    # "InstantX/FLUX.1-dev-controlnet-canny"
    "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/InstantX/FLUX.1-dev-controlnet-canny_main"
    , torch_dtype=torch.bfloat16
)

device = "cuda:0"

#%%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-schnell_main"
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)
quantize_(
    transformer
    , int8_weight_only() 
    , device = "cuda:0"      # quantize using GPU to accelerate the speed
)

#%%
pipe = FluxControlNetInpaintPipeline.from_pretrained(
    #"black-forest-labs/FLUX.1-schnell"
    model_path
    , transformer = transformer
    , controlnet = controlnet
    , torch_dtype = torch.bfloat16
)
pipe.enable_model_cpu_offload(gpu_id = 0)

#%%
control_image = load_image(
    "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg"
)
display(control_image)
init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
)
display(init_image)

mask_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
)
display(mask_image)

#%%
# prompt = "A girl holding a sign that says InstantX"
prompt = "A girl in city, 25 years old, cool, futuristic"
image = pipe(
    prompt,
    # image                           = init_image,
    # mask_image                      = mask_image,
    control_image                   = control_image,
    control_guidance_start          = 0.2,
    control_guidance_end            = 0.8,
    controlnet_conditioning_scale   = 0.7,
    strength                        = 0.7,
    num_inference_steps             = 28,
    guidance_scale                  = 3.5,
).images[0]
image.save("flux_controlnet_inpaint.png")