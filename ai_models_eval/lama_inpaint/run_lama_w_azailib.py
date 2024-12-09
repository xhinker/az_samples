#%%
from azailib.image_model_pipes import LAMAInpaintPipe

checkpoint_path = "/home/andrewzhu/storage_1t_1/github_repos/lama/big-lama/models/best.ckpt"
lama_pipe = LAMAInpaintPipe(checkpoint_path=checkpoint_path)

#%%
image_path      = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/lama_inpaint/test_images/image_w_watermark.png"
mask_path       = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/lama_inpaint/test_images/image_w_watermark_mask.png"
output_img = lama_pipe.predict(
    input_img_path=image_path
    , input_mask_path=mask_path
)
output_img