#%%
from azailib.image_model_pipes import RembgPipe
rembg_pipe = RembgPipe()
image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/dress6.png"

#%%
output = rembg_pipe.remove_background(
    image_or_path=image_path
    , height=768*2
    , width=512*2
)
output