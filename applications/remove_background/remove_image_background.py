'''
for CUDA 12.x, Install onnxruntim-gpu using
`pip install flatbuffers numpy packaging protobuf sympy`
`pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-gpu`
'''

#%%
from rembg import remove
from PIL import Image
from diffusers.utils import load_image

image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/dress6.png"
source_image = load_image(image_path)
source_image

#%%
#white_bg = Image.new("RGBA", source_image.size, (255,255,255))
black_bg = Image.new("RGBA", source_image.size, (255,255,255))
image_wo_bg = remove(source_image)
img_wo_bg = Image.alpha_composite(black_bg, image_wo_bg)
img_wo_bg

