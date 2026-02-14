#%%
from diffusers.utils import export_to_video, load_image
import numpy as np

# image_path = f"/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_{seed}.png"
# # image_path = f"/home/andrewzhu/storage_8t_4/sd_input_output/202504/flux/output_seed_{seed}.png"
# image = load_image(image_path)

# Choose a VRAM-friendly size (≈480p area). Wan’s example uses a similar compute:
max_area    = 480 * 832  # tweak upward/downward if you have headroom
# max_area    = 720 * 1024  # tweak upward/downward if you have headroom

width, height = 1736,2448
aspect      = height / width
mod         = 16
height      = (round(np.sqrt(max_area * aspect)) // mod) * mod
width       = (round(np.sqrt(max_area / aspect)) // mod) * mod

print(mod, width, height)