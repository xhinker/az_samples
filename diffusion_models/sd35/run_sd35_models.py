# %% [markdown]
# # Run Stable Diffusion 3.5 models
# 
# Ensure to have the following packages installed before running this notebook for saving time to install packages one by one.
# ```sh
# accelerate
# bitsandbytes
# protobuf
# transformers
# diffusers
# sentencepiece
# ```

# %%
# import packages
import torch
from transformers import BitsAndBytesConfig
from diffusers import (
    SD3Transformer2DModel
    ,StableDiffusion3Pipeline
)
import gc

# settings
device = "cuda:0"
prompt = "4k, best quality, beautiful 20 years girl, show hands and fingers"

def get_sd35_pipe(model_path:str):
    return StableDiffusion3Pipeline.from_pretrained(
        model_path
        , torch_dtype           = torch.bfloat16
    )

def get_sd35_quant_pipe(
    model_path:str
    , bit_type = "8bit"
):
    '''
    Function to get pipe
    '''
    if bit_type == "8bit":
        nf_config = BitsAndBytesConfig(load_in_8bit = True)
        sd35_nf = SD3Transformer2DModel.from_pretrained(
            model_path
            , subfolder             = "transformer"
            , quantization_config   = nf_config
            , torch_dtype           = torch.bfloat16
        )
    elif bit_type == "4bit":
        nf_config = BitsAndBytesConfig(
            load_in_4bit                = True
            , bnb_4bit_quant_type       = "nf4"
            , bnb_4bit_compute_dtype    = torch.bfloat16
        )
        sd35_nf = SD3Transformer2DModel.from_pretrained(
            model_path
            , subfolder           = "transformer"
            , quantization_config = nf_config
            , torch_dtype         = torch.bfloat16
        )
    
    sd35_pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path
        , transformer           = sd35_nf
        , torch_dtype           = torch.bfloat16
    )
    return sd35_pipe

def run_sd_pipe(pipe:StableDiffusion3Pipeline, prompt:str, seed:int=7):
    '''
    All models reuse this function to avoid writing duplicated code
    '''
    pipe.to(device)
    
    neg_prompt = "low quality, blur"

    image = pipe(
        prompt              = prompt
        , negative_prompt   = neg_prompt
        , generator = torch.Generator(device).manual_seed(seed)
    ).images[0]
    display(image)

    pipe.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()



# %% [markdown]
# ## Stable Diffusion 3.5 Large

# %% [markdown]
# For 24G VRAM GPU like RTX 3090 or RTX 4090, Have to quantize model to 4 or 8 bit before running.

# %% [markdown]
# ### SD35 large 8-bit quantization

# %%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/stabilityai/stable-diffusion-3.5-large_main"
# or model_path = "stabilityai/stable-diffusion-3.5-large"

pipe = get_sd35_quant_pipe(
    model_path = model_path
    , bit_type = "8bit"
)

# %% [markdown]
# Since the quantizing will move the transformer model to VRAM, no need to manually move it to CUDA, generate images and then release the VRAM.

# %%
run_sd_pipe(
    pipe        = pipe
    , prompt    = prompt
    , seed      = 7
)

# %% [markdown]
# Alas, SD 3.5 still gets problems with fingers. This 8-bit model will use around 21G VRAM, now let's see how much VRAM will be used by 4-bit model. Please restart this notebook to release the occupied VRAM before running the following code.
# 
# ### SD35 large 4-bit quantization

# %%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/stabilityai/stable-diffusion-3.5-large_main"
pipe = get_sd35_quant_pipe(
    model_path = model_path
    , bit_type = "4bit"
)

# %%
run_sd_pipe(
    pipe        = pipe
    , prompt    = prompt
    , seed      = 7
)

# %% [markdown]
# The 4-bit quantizing output a very different result compare with the 8-bit version. Face and overall aesthetics is good, but the hand and fingers, are still a mess.

# %% [markdown]
# ## Stable Diffusion 3.5 Large Turbo

# %% [markdown]
# ### SD35 Large Turbo 8-bit
# Let's see the output of SD 3.5 large turbo version, in 8bit.

# %%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/stabilityai/stable-diffusion-3.5-large-turbo_main"
pipe = get_sd35_quant_pipe(
    model_path = model_path
    , bit_type = "8bit"
)

# %%
run_sd_pipe(
    pipe        = pipe
    , prompt    = prompt
    , seed      = 7
)

# %% [markdown]
# The 8bit result is bad, really really bad. How about 4bit? 

# %% [markdown]
# ### SD35 Large Turbo 4-bit

# %%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/stabilityai/stable-diffusion-3.5-large-turbo_main"
pipe = get_sd35_quant_pipe(
    model_path = model_path
    , bit_type = "4bit"
)

# %%
run_sd_pipe(
    pipe        = pipe
    , prompt    = prompt
    , seed      = 7
)

# %% [markdown]
# Really not very good.

# %% [markdown]
# ## Stable Diffusion 3.5 Medium

# %% [markdown]
# ### SD 35 Medium bf16

# %%
model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/stabilityai/stable-diffusion-3.5-medium_main"
pipe = get_sd35_pipe(model_path=model_path)

# %%
run_sd_pipe(
    pipe = pipe
    , prompt = prompt
    , seed = 7
)

# %% [markdown]
# Face is acceptable, but fingers, hmmm

# %% [markdown]
# ### SD 35 Medium 8-bit

# %%
model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/stabilityai/stable-diffusion-3.5-medium_main"
pipe = get_sd35_quant_pipe(model_path=model_path, bit_type="8bit")
run_sd_pipe(
    pipe = pipe
    , prompt = prompt
    , seed = 7
)

# %% [markdown]
# 8bit is Even worse.

# %% [markdown]
# ### SD 35 Medium 4-bit

# %%
model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/stabilityai/stable-diffusion-3.5-medium_main"
pipe = get_sd35_quant_pipe(model_path=model_path, bit_type="4bit")
run_sd_pipe(
    pipe = pipe
    , prompt = prompt
    , seed = 7
)

# %% [markdown]
# This is Unacceptable.


