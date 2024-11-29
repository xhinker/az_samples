'''
only good for 1024 x 1024. not very good 
'''

#%%
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

model_path = "/home/andrewzhu/storage_1t_1/sdxl_models/sd_xl_base_1.0"
# model_path  = "/home/andrewzhu/storage_1t_1/sdxl_models/fudukiMix_v15"
device      = "cuda:0"

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    model_path
    , torch_dtype       = torch.float16
    #, variant           = "fp16"
    , use_safetensors   = True
)
pipe.to(device)

#%%
img_url     = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url    = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
prompt      = "A cat sitting on a bench"

# img_url     = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/images/jeans.png'
# mask_url    = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/mask.png'
# prompt      = """\
# A woman with blonde hair standing like a model. A woman wearing yellow t-shirt.
# """

init_image = load_image(img_url).convert("RGB")
display(init_image)
(w, h) = init_image.size

mask_image = load_image(mask_url).convert("RGB")
display(mask_image)

#%%
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
image = pipe(
    prompt                  = prompt
    , image                 = init_image
    , mask_image            = mask_image
    , guidance_scale        = 12.0
    , width                 = 1024 #w
    , height                = 1024 #h
    , num_inference_steps   = 40
    , strength              = 0.9
    , generator             = torch.Generator('cuda:0').manual_seed(124)
).images[0]
image