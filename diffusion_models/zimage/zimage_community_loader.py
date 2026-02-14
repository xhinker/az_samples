#%%
from diffusers import DiffusionPipeline

local_pipeline_path = "/mnt/data_2t_3/github_repos/ComfyUI/models/checkpoints/zimage/redcraftRedzimageUpdatedDEC03_redzimage15AIO.safetensors"

# Load the entire pipeline locally. Diffusers handles loading the components internally.
pipeline = DiffusionPipeline.from_pretrained(
    local_pipeline_path,
    custom_pipeline="Tongyi-MAI/Z-Image-Turbo", # specify the pipeline type if needed
    use_safetensors=True
)

# text_encoder = pipeline.text_encoder
# tokenizer = pipeline.tokenizer