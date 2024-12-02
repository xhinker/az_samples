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
#image_path  = 'source_images/face.png'
#prompt      = "sunny day, with flowers in the background"

image_path  = 'https://preview.redd.it/flux-dev-outpainting-v0-9ehgpg8yzg4e1.jpg?width=1022&format=pjpg&auto=webp&s=deaf195c8894a897c007618aebb47f2c02e53ed5'
prompt      = ""
from azailib.image_tools import generate_outpaint_image_mask

outpaint_pixel = 400
image, mask = generate_outpaint_image_mask(
    input_image             = image_path
    , top_expand            = outpaint_pixel
    , right_expand          = outpaint_pixel
    , bottom_expand         = outpaint_pixel
    , left_expand           = outpaint_pixel
    , original_image_scale  = 0.6
)

(w,h) = mask.size
print(w, h)

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