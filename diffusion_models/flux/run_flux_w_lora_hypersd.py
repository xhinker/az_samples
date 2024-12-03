#%%
import torch
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

#%%
model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
gpu_id      = 0
device      = f"cuda:{gpu_id}"

# load transformer model and quantize to int8
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)
quantize_(
    transformer
    , int8_weight_only()
    , device = device
)

#%%
pipe = DiffusionPipeline.from_pretrained(
    model_path
    , transformer = transformer
    , torch_dtype = torch.bfloat16
)

pipe.enable_model_cpu_offload(gpu_id=gpu_id)

#%% load lora 
# lora_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/UltraRealPhoto.safetensors"
# pipe.load_lora_weights(lora_path)

lora_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/Hyper-FLUX.1-dev-8steps-lora.safetensors"
pipe.load_lora_weights(
    lora_path
    , adapter_name = "hypersd"
)

#%%
lora_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/AsirAsianPhotographyflux.safetensors"
pipe.load_lora_weights(
    lora_path
    , adapter_name = "asian"
)

#%%
pipe.set_adapters("hypersd", 0.125) 
pipe.set_adapters("asian", 0.4) 

#%%
# pipe.fuse_lora(lora_scale=0.125)

#%%
prompt = """
Fashion portrait photo of thin young japan woman, 25yo, 
with light green almond-shaped eyes and narrow lips with an arrogant attitude. 
She has tousled auburn emo scene hair and emo makeup featuring shades of white and mascara with glossy lipstick, 
wearing sleeveless faux leather graphic tee in combination with a beaded necklace. 
Set on an european castle visit with authentic medieval walls, 
towers and stone arches with a cobbled path leading through, 
on a sunny rainy sunrise in summer with a shy smile. With artificial split lighting, 
shallow depth of field, dynamic angle
"""
prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
    pipe        = pipe
    , prompt    = prompt
)

#%%
seed = 11
# pipe.unload_lora_weights()

for i in range(seed, seed + 1):
    print(i)
    image = pipe(
        prompt_embeds               = prompt_embeds
        , pooled_prompt_embeds      = pooled_prompt_embeds
        #prompt                      = prompt # you can also provide the prompt here directly
        , width                     = 1024 #1024#1680
        , height                    = 1024 #1680#1024
        , num_inference_steps       = 16 #24
        , generator                 = torch.Generator().manual_seed(i)
        , guidance_scale            = 3.5
        #, joint_attention_kwargs    = {"scale": 0.125}
    ).images[0]
    display(image)