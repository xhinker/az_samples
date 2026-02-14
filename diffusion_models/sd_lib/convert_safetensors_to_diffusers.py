#%%
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

# ckpt_checkpoint_path = r"/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/sdxl_checkpoints/pornmaster_asianSdxlV1VAE.safetensors"
# target_path = r"/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/sdxl_checkpoints/pornmaster_asianSdxlV1VAE"
ckpt_checkpoint_path = r"/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/sdxl_checkpoints/divingIllustriousReal_v20VAE.safetensors"
target_path = r"/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/sdxl_checkpoints/divingIllustriousReal_v20VAE"

pipe = download_from_original_stable_diffusion_ckpt(
    ckpt_checkpoint_path
    , from_safetensors  = True
    , device            = "cuda:0"
)
pipe.save_pretrained(target_path)