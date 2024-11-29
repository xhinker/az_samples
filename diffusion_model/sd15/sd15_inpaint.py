#%%
import torch
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from diffusers.utils import load_image

model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/CompVis/stable-diffusion-v1-4_main"

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path
    , torch_dtype       = torch.float16
    , safety_checker    = None
).to("cuda:0")

#%%
# img_url     = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url    = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
# prompt      = "A dog sitting on a bench"


img_url     = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/images/jeans.png'
mask_url    = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/mask.png'
prompt      = """\
A woman wearing yellow t-shirt.
"""

init_image = load_image(img_url).convert("RGB")
display(init_image)
(w, h) = init_image.size
print(w,h)

mask_image = load_image(mask_url).convert("RGB")
display(mask_image)

#%%
inpaint_pipe.scheduler = EulerDiscreteScheduler.from_config(inpaint_pipe.scheduler.config)
image = inpaint_pipe(
    prompt                  = prompt
    , image                 = init_image
    , mask_image            = mask_image
    , guidance_scale        = 16.0
    , width                 = w
    , height                = h
    , num_inference_steps   = 40
    , strength              = 0.95
    , generator             = torch.Generator('cuda:0').manual_seed(123)
).images[0]
image