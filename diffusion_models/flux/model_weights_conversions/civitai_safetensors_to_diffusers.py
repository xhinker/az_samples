#%%
from sd_embed.conversion_tools import Flux1Convertor
convertor = Flux1Convertor()

#%%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_checkpoints/atomixFLUXUnet_v10.safetensors"
convertor.convert_to_diffuses(input_safetensor_file_path=model_path)