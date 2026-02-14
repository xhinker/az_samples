#%%
import torch
from diffusers import GGUFQuantizationConfig
from diffusers.models import ZImageTransformer2DModel

model_path = '/mnt/data_2t_3/github_repos/ComfyUI/models/checkpoints/zimage/zImage_turbo.safetensors'

model = ZImageTransformer2DModel.from_single_file(
    model_path,
    # quantization_config     = GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype             = torch.bfloat16,
)

#%% load VAE
from diffusers import AutoencoderKL
vae_path = "/mnt/data_2t_3/github_repos/ComfyUI/models/vae/zimage/zImage_vae.safetensors"
vae = AutoencoderKL.from_single_file(vae_path)

#%% load text encoder
from transformers import PreTrainedModel

text_encoder_path = '/mnt/data_2t_3/github_repos/ComfyUI/models/text_encoders/zimage/zImage_textEncoder.safetensors'

text_encoder = PreTrainedModel.from_pretrained(
    pretrained_model_name_or_path = text_encoder_path,
    torch_dtype                   = torch.bfloat16
)