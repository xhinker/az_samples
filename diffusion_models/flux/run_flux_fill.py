'''
Update diffusers to the newest version
`pip install -U diffusers`

This inpaint model could be the best so far. 
Working well with bright color, but not dark color, if the source object color is black, need to hight light it.
'''

#%%
import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
from diffusers.utils import load_image

model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Fill-dev_main"
device = "cuda:0"

#%%
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)
quantize_(
    transformer
    , int8_weight_only() 
    , device = device # quantize using GPU to accelerate the speed
)

#%%
pipe = FluxFillPipeline.from_pretrained(
    model_path
    , transformer = transformer
    , torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

# #%%
# image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
# display(image)
# mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")
# display(mask)
# prompt = "a white paper cup"

#%%
# image_path  = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/images/jeans.png'
# mask_path   = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/mask.png'
# prompt      = """\
# bright color suit with tie
# """

# image_path  = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_model/flux/source_images/suit_w_bag.png'
# mask_path   = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_model/flux/source_images/suit_w_bag_mask.png'
# prompt      = """\

# """
image_path  = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/image_w_watermark2_s1.png'
mask_path   = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/image_w_watermark2_s1_mask.png'
prompt      = ""

# Set image path , mask path and prompt
# image_path  = 'https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket.png'
# mask_path   = 'https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket_mask.jpeg'
# prompt      = 'a person wearing a white shoe, carrying a white bucket with text "FLUX" on it'

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