#%%
from azailib.image_model_pipes import (
    GroundingDinoPipeline
    , SAMModelPipe
)

groundingdino_model_path = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
sam2_checkpoint = "/home/andrewzhu/storage_1t_1/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt"

dino_pipe = GroundingDinoPipeline(
    checkpoint_path_or_id = groundingdino_model_path
)

#%%
# load SAM2 pipe
sam2_pipe = SAMModelPipe(
    checkpoint_path_or_id = sam2_checkpoint
)

#%%
image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/jeans.png"
target_prompt = "black shirt"

anotated_img, boxes = dino_pipe.predict(
    image_path = image_path
    , prompt = target_prompt
)
display(anotated_img)
print(boxes)

#%%
masks = sam2_pipe.get_masks(
    image_or_path = image_path
    , xyxy_boxes = boxes
    , show_middle_masks = True
    , dilate_margin = 10
) 
masks

#%%
masks.save("output_images/jeans_mask.png")
