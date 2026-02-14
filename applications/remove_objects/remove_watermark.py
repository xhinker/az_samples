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

#%%
# load SAM2 pipe
sam2_pipe = SAMModelPipe(
    checkpoint_path_or_id = sam2_checkpoint
)

#%%
# load LAMA inpaint
lama_pipe = LAMAInpaintPipe(checkpoint_path=lama_checkpoint_path)

#%%
# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/long_dress10_watermark.png"
# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/applications/remove_objects/images_with_watermark/Weixin Image_20250506213434.jpg"
# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/applications/remove_objects/images_with_watermark/Weixin Image_20250506213440.jpg"
# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/applications/remove_objects/images_with_watermark/Weixin Image_20250506213444.jpg"
# image_path = "stage3.png"
image_path = "image1_stage2.png"
target_prompt = "text watermark"
# target_prompt = "text"

anotated_img, boxes = dino_pipe.predict(
    image_path          = image_path
    , prompt            = target_prompt
    , box_threshold     = 0.07
    , text_threshold    = 0.1
)
display(anotated_img)
print(boxes)

#%%
boxes = boxes[:1]
boxes

#%%
masks = sam2_pipe.get_masks(
    image_or_path       = image_path
    , xyxy_boxes        = boxes
    , show_middle_masks = True
    , dilate_margin     = 7#10
) 
masks_path = "mask.png"
masks.save(masks_path)
masks

#%%
output_img = lama_pipe.predict(
    input_img_path      = image_path
    , input_mask_path   = masks_path
)
# output_img.save("/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/local_tests/images/long_dress10_no_watermark.png")
display(output_img)

#%%
output_img.save("image1_stage2.png")