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
# image_path  = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/suit_w_bag.png"
image_path  = "/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/output_seed_945.png"

init_image  = load_image(image_path)
# resize image
upscale_times = 1.5
w,h = init_image.size
print(h,w)
new_height = int(h*upscale_times)
new_width = int(w*upscale_times)

input_img   = resize_img(
    image_or_path   = image_path
    , width         = new_width
    , height        = new_height
)
# print(input_img.size)
input_img

#%%
prompt = """\
8k, raw photo, extreme high resolution
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
    , num_inference_steps   = 30
    , strength              = 0.25
    , guidance_scale        = 3.5
    , generator             = torch.Generator("cuda:0").manual_seed(19)
).images[0]
display(image)