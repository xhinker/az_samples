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

ootd_pipe = OOTDPipe(
    body_pose_checkpoint_path           = body_pose_checkpoint_path
    , humanparsing_atr_checkpoint_path  = humanparsing_atr_checkpoint_path
    , humanparsing_lip_checkpoint_path  = humanparsing_lip_checkpoint_path
    , gpu_id=1
)

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
# image_path = '/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/run/examples/image-4.png'
image_path = '/home/andrewzhu/storage_8t_4/sd_input_output/20231206/fudukiMix_v15_0.5_0.2_0.5_korean_upscale_1.75_1346.png'
load_image(image_path)

#%% generate mask using ootd
mask = ootd_pipe.get_mask(
    image_or_path   = image_path
    # , body_position = "lower_body_w_shoe"  #"dresses"
    # , body_position = "dresses_w_shoe"  #"dresses"
    # , body_position = "upper_body"  #"dresses"
    # , body_position = "dresses"  #"dresses"
    , body_position = "whole_body_except_head"  #"dresses"
    , dilate_margin = 1
)
mask_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/model_mask.png"
mask.save(mask_path)
mask

#%% generate mask using GroundingDino and sam
# target_words = [
#     "shirt"
#     , "arms"
#     , "shoulder"
# ]

# all_boxes = []
# for prompt in target_words:
#     anotated_img, boxes = dino_pipe.predict(
#         image_path = image_path
#         , prompt = prompt
#         , box_threshold = 0.3
#     )
#     display(anotated_img)
#     print(boxes)
#     all_boxes = [*all_boxes, *boxes]
# print(all_boxes)

# # generate mask
# mask = sam2_pipe.get_masks(
#     image_or_path       = image_path
#     , xyxy_boxes        = all_boxes
#     , show_middle_masks = True
#     , dilate_margin     = 10
# ) 
# mask_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/model_mask.png"
# mask.save(mask_path)
# mask

#%% scale image and mask
# Load image and mask
# base_image = load_image(image_path)#.convert("RGB")#.resize(size)
base_image = scale_img(
    img_path        = image_path
    , upscale_times = 0.65
)
(w,h) = base_image.size
display(base_image)
print(w,h)
# mask = load_image(mask_path)#.convert("RGB")#.resize(size)
mask = resize_img(image_or_path=mask_path,width=w, height=h)
display(mask)

#%% load dress image and remove background
# dress_image_path = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/long_dress10_no_watermark.png'
dress_image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/short_dress8.png"

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

#%% get the box
# anotated_img, boxes = dino_pipe.predict(
#     image_path = resized_dress_path
#     , prompt = "dress" 
# )
# display(anotated_img)
# print(boxes)

# # get the boxed part
# dress_image = extract_objects_using_xyxy_boxes(
#     img = dress_image
#     , boxes = boxes
# )
# dress_image

#%% generate mask
# dress_mask = sam2_pipe.get_masks(
#     image_or_path       = dress_image_path
#     , xyxy_boxes        = boxes
#     , show_middle_masks = True
#     , dilate_margin     = 0
# ) 
# mask_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/diffusion_models/flux/source_images/dress_mask.png"
# dress_mask.save(mask_path)
# dress_mask = resize_img(
#     img_path = mask_path
#     , width = w
#     , height = h
# )
# dress_mask

# dress_cloth = extract_object_on_white_background(
#     img = dress_image
#     , mask_img = dress_mask
# )
# dress_cloth


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
    , num_inference_steps   = 40
    , max_sequence_length   = 512
    , generator             = torch.Generator("cpu").manual_seed(5)
    , **pipe_prior_output
).images[0]
image