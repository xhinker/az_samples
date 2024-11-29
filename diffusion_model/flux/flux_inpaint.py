'''
Need verify
'''

#%%
import torch
from diffusers import FluxControlNetInpaintPipeline
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image
controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-controlnet-canny", torch_dtype=torch.float16
)
pipe = FluxControlNetInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.to("cuda")
control_image = load_image(
    "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg"
)
init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
)
mask_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
)
prompt = "A girl holding a sign that says InstantX"
image = pipe(
    prompt,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
    control_guidance_start=0.2,
    control_guidance_end=0.8,
    controlnet_conditioning_scale=0.7,
    strength=0.7,
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("flux_controlnet_inpaint.png")