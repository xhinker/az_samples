#%%
import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from torchao.quantization import quantize_, int8_weight_only
from azailib.image_tools import (
    resize_img
    , scale_img
    , concatenate_images_left_right
    , extend_mask_left
    , extract_object_on_white_background
    , extract_objects_using_xyxy_boxes
)

#%%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Kontext-dev_main"

# load transformer model and quantize to int8
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder 	= "transformer"
    , torch_dtype 	= torch.bfloat16
)
quantize_(transformer, int8_weight_only())

#%%
pipe = FluxKontextPipeline.from_pretrained(
    model_path
    , transformer 	= transformer
    , torch_dtype 	= torch.bfloat16
)
pipe.enable_model_cpu_offload()


#%% image modify
# input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
# input_image = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1160.png")
# input_image = load_image("/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux_kontext/images/Weixin Image_20250506213440.jpg")
# input_image = load_image("/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux_kontext/images/image.png")
input_image = load_image("/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/framepack/sample_images/image.png")
display(input_image)
w, h = input_image.size
print(w,h)

# prompt = "the scene is exactly the same, but add floral headwear with colorful flowers in her hair, and an elegant white lace collar around her neck"
# prompt = "remove words and watermark"
# prompt = "remove people in the selfie photo"
# prompt = "remove all people in the background from the selfie photo"
prompt = "the scene is exactly the same, the same pattern, but change the dress color to flowery gold yellow"

image = pipe(
	image           	= input_image
	, prompt 			= prompt
	, guidance_scale 	= 2.5
	, height 			= h
    , width 			= w
).images[0]
display(image)

#%% remove background objects
input_image = load_image("/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux_kontext/images/image.png")
display(input_image)
w, h = input_image.size
print(w,h)

# prompt = "remove people in the selfie photo"
prompt = "remove all people in the background from the selfie photo"

image = pipe(
	image           	= input_image
	, prompt 			= prompt
	, guidance_scale 	= 2.5
	, height 			= h
    , width 			= w
).images[0]
display(image)

#%%


#%% merge images
input_image_1 = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1104.png")
input_image_2 = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1108.png")
input_image = concatenate_images_left_right(input_image_1, input_image_2)

display(input_image)
w, h = input_image.size
print(w,h)

# prompt = "the scene is exactly the same, but add floral headwear with colorful flowers in her hair, and an elegant white lace collar around her neck"
# prompt = "place both sexy girl together in one scene where they are holding hands and standing together, both face front"
prompt = "place both sexy girl together in one scene where they are holding hands and standing together, both face front"

image = pipe(
	image           	= input_image
	, prompt 			= prompt
	, guidance_scale 	= 2.5
    , num_inference_steps = 30
	, height 			= h
    , width 			= w
	, generator         = torch.Generator('cpu').manual_seed(3)
).images[0]
display(image)

#%% VTO 
input_image_1 = load_image("https://shop.simon.com/cdn/shop/files/27683494fe4140139c23aa9633ae2604_1800x1800.jpg?v=1725423887")
input_image_2 = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1162.png")
input_image = concatenate_images_left_right(input_image_1, input_image_2)
display(input_image)

# input_image_1 = resize_img(input_image_1, height=1200, width=864)
# input_image_2 = resize_img(input_image_2, height=1200, width=864)

# display(input_image_1)
# display(input_image_2)
w, h = input_image.size
print(w,h)

#%%
prompt = """Dress the person in the right image with the clothing item from the right girl, 
ensuring the clothing fits naturally, the pose remains the same, and the background is unchanged.
"""

image = pipe(
	image           	    = input_image
	, prompt 			    = prompt
	, guidance_scale 	    = 2.5
    , num_inference_steps   = 28
	, height 			    = h
    , width 			    = w
	, generator             = torch.Generator('cpu').manual_seed(4)
).images[0]
display(image)

#
#%% hold hand bag - not working
input_image_1 = load_image("https://shop.simon.com/cdn/shop/files/27683494fe4140139c23aa9633ae2604_1800x1800.jpg?v=1725423887")
input_image_2 = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1162.png")
input_image = concatenate_images_left_right(input_image_1, input_image_2)
display(input_image)

# input_image_1 = resize_img(input_image_1, height=1200, width=864)
# input_image_2 = resize_img(input_image_2, height=1200, width=864)

# display(input_image_1)
# display(input_image_2)
w, h = input_image.size
print(w,h)

prompt = "merge the bag and women, so that the image show the beautiful woman's hand hold the bag from the left"

image = pipe(
	image           	    = input_image
	, prompt 			    = prompt
	, guidance_scale 	    = 7#2.5
    , num_inference_steps   = 28
	, height 			    = h
    , width 			    = w
	, generator             = torch.Generator('cpu').manual_seed(4)
).images[0]
display(image)