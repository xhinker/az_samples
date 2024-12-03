'''
The Flux.1 Img2Img sample code
'''

#%%
import torch
from azailib.sd_pipe_loaders import load_flux1_img2img_8bit_pipe
from diffusers.utils import load_image
from azailib.image_tools import resize_img
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
pipe = load_flux1_img2img_8bit_pipe(checkpoint_path_or_id = model_path)

#%% load image and resize
image_path  = "source_images/suit_w_bag.png"
init_image  = load_image(image_path)
# resize image
input_img   = resize_img(img_path=image_path, upscale_times=0.75)
print(input_img.size)
input_img

#%%
prompt = """\
She dresses leather jacket, wear distressed (blue jeans), wear yellow shoe, hand hold red bag.
"""
prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
    pipe        = pipe
    , prompt    = prompt
)

image = pipe(
    prompt_embeds           = prompt_embeds
    , pooled_prompt_embeds  = pooled_prompt_embeds
    #prompt                 = prompt
    , image                 = input_img
    , height                = input_img.size[1]
    , width                 = input_img.size[0]
    , num_inference_steps   = 50
    , strength              = 0.85
    , guidance_scale        = 3.5
    , generator             = torch.Generator("cuda:0").manual_seed(19)
).images[0]
display(image)