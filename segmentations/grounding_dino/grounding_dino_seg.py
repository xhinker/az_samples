'''
To use Grounding Dino
1. clone the repo
2. Install torch
3. `pip install -r requirements.txt`
4. `pip install groundingdino-py`
'''

#%%
from groundingdino.util.inference import load_model, predict, annotate
from groundingdino.util.inference import load_image as load_image_dino
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import sys
from pathlib import Path
base_path = Path(__file__).resolve().parent.parent
sys.path.append(str(base_path))
from segmentations.sematic_seg.common_tools import *

device = "cuda:0"

def show_cv2_img(image_data):
    img_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    display(pil_img)

#%%
model_py_path       = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
model_weight_path   = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "../04-06-segment-anything/weights/groundingdino_swint_ogc.pth")
model               = load_model(
    model_py_path
    , model_weight_path
    , device        = device
)

#%%
# IMAGE_PATH          = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/.asset/cat_dog.jpeg"
# TEXT_PROMPT         = "chair . person . dog ."
# IMAGE_PATH          = "/home/andrewzhu/storage_1t_1/az_git_folder/azcode/az_projects/model_tests/.model_test/image-2.png"
IMAGE_PATH          = "./images/jeans.png"
# TEXT_PROMPT         = "black shirt"
TEXT_PROMPT         = "hair"

BOX_TRESHOLD        = 0.4
TEXT_TRESHOLD       = 0.4

image_source, image = load_image_dino(IMAGE_PATH)

#%% 
boxes, logits, phrases = predict(
    model               = model
    , image             = image
    , caption           = TEXT_PROMPT
    , box_threshold     = BOX_TRESHOLD
    , text_threshold    = TEXT_TRESHOLD
    , device            = device
)

annotated_frame = annotate(
    image_source    = image_source
    , boxes         = boxes
    , logits        = logits
    , phrases       = phrases
)
#cv2.imwrite("annotated_image.jpg", annotated_frame)
show_cv2_img(annotated_frame)

#%%
from groundingdino.util import box_ops
from torchvision.ops import box_convert
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
xyxy = [int(i) for i in xyxy[0]]
xyxy

#%%
checkpoint = "/home/andrewzhu/storage_1t_1/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt"
# the following format is required, the original writer did it wrong for full absolute path
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda:0"
predictor = load_sam2_model(
    checkpoint_path     = checkpoint
    , model_cfg_path    = model_cfg
    , device            = device
)