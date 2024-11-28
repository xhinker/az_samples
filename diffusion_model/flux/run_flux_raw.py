'''
The code run Flux.1 with more than 24G Vram, RTX 4090, can run it with 1024x1024
'''

#%%
import torch
from diffusers import FluxPipeline

model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"

pipe = FluxPipeline.from_pretrained(
    model_path
    , torch_dtype=torch.bfloat16
)

#save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.enable_model_cpu_offload() 

#%%
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale      = 3.5,
    output_type         = "pil",
    num_inference_steps = 20,
    max_sequence_length = 512,
    generator           = torch.Generator("cuda").manual_seed(0)
).images[0]
image.save("flux-schnell.png")
image