'''
Update diffusers to the newest version
`pip install -U diffusers`

Install azailib
`pip install -U git+https://github.com/xhinker/azailib.git`
'''

#%%
import torch
from azailib.sd_pipe_loaders import load_flux1_fill_8bit_pipe
from diffusers.utils import load_image

model_path              = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Fill-dev_main"
pipe_device             = "cuda:0"

pipe = load_flux1_fill_8bit_pipe(
    checkpoint_path_or_id       = model_path
    , pipe_device               = pipe_device
)

#%%
image_path  = 'source_images/outpaint_image.png'
mask_path   = 'source_images/outpaint_image_mask.png'
prompt      = ""
from azailib.image_tools import generate_outpaint_image_mask

image, mask = generate_outpaint_image_mask(
    input_image = image_path
)

(w,h) = mask.size

display(image)
display(mask)

#%%
image = pipe(
    prompt                  = prompt
    , image                 = image
    , mask_image            = mask
    , height                = h
    , width                 = w
    , guidance_scale        = 30 
    , num_inference_steps   = 30
    , max_sequence_length   = 512
    , generator             = torch.Generator("cpu").manual_seed(4)
).images[0]
image