'''
1. Use grounding dino to get the target boxes
2. Use Sam2 to get the target mask
'''

#%% load required packages
import torch
from groundingdino.util.inference import load_model as dino_load_model
from groundingdino.util.inference import predict as dino_predict
from groundingdino.util.inference import annotate as dino_annotate
from groundingdino.util.inference import load_image as dino_load_image
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from diffusers.utils import load_image

from common_tools import (
    load_sam2_model
    , show_cv2_img
    , get_xyxy_boxes
    , show_masks
    , get_mask_img
)

dino_device = "cuda:1"
sam2_device = "cuda:1"

#%% load grounding dino model
model_py_path       = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
model_weight_path   = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
model               = dino_load_model(
    model_py_path
    , model_weight_path
    , device        = dino_device
)
BOX_TRESHOLD        = 0.3
TEXT_TRESHOLD       = 0.3


#%% start sam2 to get the mask
checkpoint = "/home/andrewzhu/storage_1t_1/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt"
# the following format is required, the original writer did it wrong for full absolute path
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# load up sam2 model
# any benefit of doing this?
# torch.autocast(device, dtype=torch.bfloat16).__enter__()
sam2_predictor = load_sam2_model(
    checkpoint_path     = checkpoint
    , model_cfg_path    = model_cfg
    , device            = sam2_device
)

#%% get boxes using dino 
# IMAGE_PATH          = "./images/jeans.png"
IMAGE_PATH          = "./images/one_truck.png"
# IMAGE_PATH          = "/home/andrewzhu/storage_1t_1/az_git_folder/azcode/az_projects/model_tests/.model_test/image-2.png"
TEXT_PROMPT         = """
car rearview mirror
"""
image_source, image = dino_load_image(IMAGE_PATH)

boxes, logits, phrases = dino_predict(
    model               = model
    , image             = image
    , caption           = TEXT_PROMPT
    , box_threshold     = BOX_TRESHOLD
    , text_threshold    = TEXT_TRESHOLD
    , device            = dino_device
)

annotated_frame = dino_annotate(
    image_source    = image_source
    , boxes         = boxes
    , logits        = logits
    , phrases       = phrases
)
#cv2.imwrite("annotated_image.jpg", annotated_frame)
show_cv2_img(annotated_frame)

xyxy_boxes = get_xyxy_boxes(boxes, image_source)
print(xyxy_boxes)

#%% get mask using detected boxes
image           = load_image(IMAGE_PATH)
display(image)
sam2_predictor.set_image(image)

input_box = np.array(xyxy_boxes[0])

masks, scores, _ = sam2_predictor.predict(
    point_coords    = None,
    point_labels    = None,
    box             = input_box[None, :],
    multimask_output=False,
)
show_masks(image, masks, scores, box_coords=input_box)

pil_mask = get_mask_img(masks[0])
pil_mask