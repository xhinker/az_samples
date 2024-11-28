#%%
import torch
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
    , torch_dtype       = torch.float16
    , safety_checker    = None
).to("cuda:0")