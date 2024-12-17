#%%
from azailib.image_model_pipes import (
    GroundingDinoPipeline
    , SAMModelPipe
    , LAMAInpaintPipe
)

groundingdino_model_path    = "/home/andrewzhu/storage_1t_1/github_repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
sam2_checkpoint             = "/home/andrewzhu/storage_1t_1/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt"
lama_checkpoint_path        = "/home/andrewzhu/storage_1t_1/github_repos/lama/big-lama/models/best.ckpt"

# load dino pipe
dino_pipe = GroundingDinoPipeline(
    checkpoint_path_or_id = groundingdino_model_path
)

# load SAM2 pipe
sam2_pipe = SAMModelPipe(
    checkpoint_path_or_id = sam2_checkpoint
)

# load LAMA inpaint
lama_pipe = LAMAInpaintPipe(checkpoint_path=lama_checkpoint_path)

#%%
image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/long_dress10_watermark.png"
target_prompt = "watermark"

anotated_img, boxes = dino_pipe.predict(
    image_path          = image_path
    , prompt            = target_prompt
    , box_threshold     = 0.26
    , text_threshold    = 0.26
)
display(anotated_img)
print(boxes)

#%%
masks = sam2_pipe.get_masks(
    image_or_path = image_path
    , xyxy_boxes        = boxes
    , show_middle_masks = True
    , dilate_margin     = 10
) 
masks_path = "mask.png"
masks.save(masks_path)
masks

#%%
output_img = lama_pipe.predict(
    input_img_path      = image_path
    , input_mask_path   = masks_path
)
output_img.save("/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/long_dress10_no_watermark.png")
display(output_img)