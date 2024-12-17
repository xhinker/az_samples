'''
Update diffusers to the newest version
`pip install -U diffusers`

Install azailib
`pip install -U git+https://github.com/xhinker/azailib.git`

Combine fill and redux, we can easily build a cloth transfer application. 
'''

#%%
import torch
from diffusers import FluxPriorReduxPipeline
from azailib.sd_pipe_loaders import load_flux1_fill_8bit_pipe
from diffusers.utils import load_image
from azailib.image_model_pipes import (
    GroundingDinoPipeline
    , SAMModelPipe
    , RembgPipe
)
from azailib.image_tools import (
    resize_img
    , scale_img
    , concatenate_images_left_right
    , extend_mask_left
    , extract_object_on_white_background
    , extract_objects_using_xyxy_boxes
)
from azailib.image_model_pipes import OOTDPipe

flux_model_path             = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Fill-dev_main"
groundingdino_model_path    = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
sam2_checkpoint             = "/home/andrewzhu/storage_1t_1/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt"
redux_model_path            = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Redux-dev_main"
device                      = "cuda:0"

body_pose_checkpoint_path           = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/openpose/ckpts/body_pose_model.pth"
humanparsing_atr_checkpoint_path    = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/humanparsing/parsing_atr.onnx"
humanparsing_lip_checkpoint_path    = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/humanparsing/parsing_lip.onnx"

#%%
rembg_pipe = RembgPipe()

pipe = load_flux1_fill_8bit_pipe(
    checkpoint_path_or_id       = flux_model_path
    , pipe_gpu_id               = 0
)

dino_pipe = GroundingDinoPipeline(
    checkpoint_path_or_id   = groundingdino_model_path
    , gpu_id                = 1
)

sam2_pipe = SAMModelPipe(
    checkpoint_path_or_id   = sam2_checkpoint
    , gpu_id                = 1
)

pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    redux_model_path
    , torch_dtype = torch.bfloat16
).to(device)

#%% generate boxes for base image
image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/house_deck.png"
input_image = load_image(image_path)
input_image

#%%
# from azailib.image_tools import get_right_half_image

# deck_image = get_right_half_image(pil_image=input_image)
# display(deck_image)
# deck_image.save(image_path)

#%% generate mask using GroundingDino and sam
# target_words = [
#     "glass slide door"
#     ,"red deck"
# ]

# all_boxes = []
# for prompt in target_words:
#     anotated_img, boxes = dino_pipe.predict(
#         image_path = image_path
#         , prompt = prompt
#         , box_threshold = 0.4
#     )
#     display(anotated_img)
#     print(boxes)
#     all_boxes = [*all_boxes, *boxes]
# print(all_boxes)

#%%
# all_boxes = [[243-100, 485-200, 433, 631]]
# # generate mask
# mask = sam2_pipe.get_masks(
#     image_or_path       = image_path
#     , xyxy_boxes        = all_boxes
#     , show_middle_masks = True
#     , dilate_margin     = 60
# ) 
mask_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/deck_mask2.png"
# mask.save(mask_path)
# mask

#%% scale image and mask
# Load image and mask
# base_image = load_image(image_path)#.convert("RGB")#.resize(size)
base_image = scale_img(
    img_path        = image_path
    , upscale_times = 1.2
)
(w,h) = base_image.size
display(base_image)
print(w,h)
# mask = load_image(mask_path)#.convert("RGB")#.resize(size)
mask = resize_img(image_or_path=mask_path,width=w, height=h)
display(mask)

#%% load dress image and remove background
dress_image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/pergola4.png"

# dress_image = rembg_pipe.remove_background(
#     image_or_path   = dress_image_path
#     , width         = w
#     , height        = h
# )

# for pure white bg
dress_image      = resize_img(
    image_or_path   = dress_image_path
    , width         = w
    , height        = h
)

resized_dress_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/dress_temp.png"
dress_image.save(resized_dress_path)
display(dress_image)


#%% generate guided image embedding
pipe_prior_output   = pipe_prior_redux(
    dress_image
)

# concate images
concate_image = concatenate_images_left_right(
    img_left = dress_image
    , img_right = base_image
)
display(concate_image)

# extend mask
ext_mask = extend_mask_left(
    mask_image = mask
)
display(ext_mask)

new_w,new_h = concate_image.size
image = pipe(
    image                   = concate_image     #base_image
    , mask_image            = ext_mask          #mask
    , height                = new_h
    , width                 = new_w
    , guidance_scale        = 30
    , num_inference_steps   = 30
    , max_sequence_length   = 512
    , generator             = torch.Generator("cpu").manual_seed(5)
    , **pipe_prior_output
).images[0]
image