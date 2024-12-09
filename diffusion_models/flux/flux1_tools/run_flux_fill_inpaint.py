'''
Update diffusers to the newest version
`pip install -U diffusers`

Install azailib
`pip install -U git+https://github.com/xhinker/azailib.git`

This inpaint model could be the best so far. 
Working well with bright color, but not dark color, if the source object color is black, need to hight light it.
'''

#%%
import torch
from azailib.sd_pipe_loaders import load_flux1_fill_8bit_pipe
from diffusers.utils import load_image

model_path              = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Fill-dev_main"

pipe = load_flux1_fill_8bit_pipe(
    checkpoint_path_or_id       = model_path
    , pipe_gpu_id               = 0
)

#%%
# image_path  = 'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png'
# mask_path   = 'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png'
# prompt      = "a white paper cup"

image_path  = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/woman_hat.png'
mask_path   = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/woman_hat_mask.png'
prompt      = """\
35 year old korean girl face
"""

# Load image and mask
image = load_image(image_path)#.convert("RGB")#.resize(size)
(w,h) = image.size
display(image)
print(w,h)
mask = load_image(mask_path)#.convert("RGB")#.resize(size)
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

#%%
# output_image = image.resize((w,h))
# output_image
image.save("/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/image_w_watermark2_s1.png")