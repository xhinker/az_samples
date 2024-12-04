'''
Code to test AWPortraitCN
'''
#%%
import torch
from azailib.sd_pipe_loaders import load_flux1_8bit_pipe
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
lora_path   = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/AWPortraitCN.safetensors"
device_id   = 0

pipe = load_flux1_8bit_pipe(checkpoint_path_or_id = model_path)

#%%
def run_pipe(
    pipe
    , prompt:str        = ""
    , seed:int          = 1
    , img_num           = 1
    , inference_steps   = 16
    , lora_adapter_name = "awportraitcn"
    , lora_weight       = 1.0
    , width             = 1024
    , height            = 1024
):
    pipe.set_adapters(lora_adapter_name, lora_weight) 
    prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
        pipe        = pipe
        , prompt    = prompt
    )
    for i in range(seed, seed + img_num):
        print(i)
        image = pipe(
            prompt_embeds               = prompt_embeds
            , pooled_prompt_embeds      = pooled_prompt_embeds
            , width                     = width
            , height                    = height
            , num_inference_steps       = inference_steps
            , generator                 = torch.Generator().manual_seed(i)
            , guidance_scale            = 3.5
        ).images[0]
        display(image)

#%% load lora
adapter_name = "awportraitcn"
pipe.load_lora_weights(
    lora_path
    , adapter_name = adapter_name
)

# pipe.unload_lora_weights()

#%% test long prompt 
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

run_pipe(pipe=pipe, prompt=prompt)
    
#%% [markdown]
# Set LoRA weight to 1.0, working well

#%% test hand
prompt = """
Fashion portrait photo of thin young chinese woman, 25yo, 
4k, best quality, beautiful 20 years girl, show hands and fingers
"""
run_pipe(
    pipe                = pipe
    , prompt            = prompt
    , seed              = 7
    , lora_weight       = 0.5
    , inference_steps   = 20
    , img_num           = 3
)

#%% [markdown]
# ## Hand test result
# 1. LoRA weight set to 1.0, hand generation capability degraded
# 2. LoRA weight set to 0.8, hand and fingers is still not good
# 3. Lower LoRA weight to 0, hands are good.
# 4. Set LoRA weight to 0.6, Six finger appears
# 5. Set LoRA weight to 0.5, finger number is good now. 
# 6. when many times generation, finger will still have problems sometimes.
# 7. Set LoRA weight to 0.4. more finger problems

#%% logic and position test
prompt = """\
cinematic photo Cinematic film still of an 18-year-old with very long blonde woman with sun-inspired macroscopic patterns. 
Bright blue eyes. Highlight concentric circles, radiating sunbeams, and floral patterns in her clothing and background, 
using a summery palette of warm yellows, oranges, and soft greens. sweet smiling. 
Aim for a serene expression and warm, diffused lighting to emphasize the delicate sun-inspired textures.
8k UHD natural lighting, raw, rich, intricate details, key visual, 
atmospheric lighting, professional, Shallow depth of field, vignette, high budget, cinemascope, vibrant, epic, gorgeous, 
film grain, grainy, cinematic photorealistic,key visual, 35mm photograph,bokeh,
4k, colorful background, concept art, 8k, dramatic lighting, 
high detail, hyper realistic, intricate, intricate sharp details, 
octane render, smooth, studio lighting, trending on artstation. 
film, highly detailed, slightly sweaty.
"""
run_pipe(
    pipe                = pipe
    , prompt            = prompt
    , seed              = 22
    , lora_weight       = 0.5
    , inference_steps   = 20
    , img_num           = 2
    , width             = 1280
    , height            = 1280
)

#%% [markdown]\
# ### Test #1 number counting
# Prompt: 
# ```
# Four apples in a squared plate, put in a plaid pattern table, the up two apples are green, the bottom two apples are red
# ```
# Code: 
# ```py
# run_pipe(
#     pipe                = pipe
#     , prompt            = prompt
#     , seed              = 10
#     , lora_weight       = 0.5
#     , inference_steps   = 16
#     , img_num           = 3
# )
# ```
# Result:
# The result are great, Flux.1 can really understand the number. 
# 
# ### Test #2 position understanding
# prompt:
# ```
# one white cat running on the left, one blue blue flying on the right side
# ```
# Result:
# Not very good, there is no bird.
# prompt： 
# ```
# Two women are negotiating a deal in a meeting room, the woman on the left with blonde blunt bangs is in white suit, the woman on the right with white curly hair is in black suit.
# ```
# 
# Overall accurate, with finger problems, increase the image resolution and inference steps do not help remedy the finger problem.
# One way to avoid the finger problem is generate more image and find the good one.  
# prompt: 
# ```
# masterpiece, realistic photo, best quality, 2 beautiful girls look at each other.
# The left one has black ponytail, wear dark blue Military uniform;
# The right one in red twintail, wear red cheongsam
# ```
# The output is not realistic photo, but more carton style, the realistic photo problem is not respected.
# All generation are similar, all in cartoon style.
#
# prompt:
# ```
# (realistic photo, nikon f4, 50mm f1.2, Fujichrome Velvia 50, bokeh), best quality, 
# masterpiece, realistic photo, best quality,
# In the left of the room, there is a delicate blue vases with pink roses,
# In the right side, A beautiful black hair girl with her eyes closed in champagne long sleeved formal dress,
# ```
# This prompt gives very good and accurate results.

# prompt:
# (realistic photo, nikon f4, 50mm f1.2, Fujichrome Velvia 50, bokeh), best quality, woman and man are chatting in a chinese restaurant,
# in the left side, a blonde long hair man with blue and white sailor uniform,
# in the right side, a black double bun hair girl, green cheongsam printed with flowers
# 
# Result:
# Fair, there will be finger problems, double bun hair is not rendered.

# ### Test #3 complex sense

# Prompt: 
# From left to right, a blonde ponytail girl in white shirt, 
# a brown curly hair girl in blue shirt printed with a bird, 
# an Asian young man with black short hair in suit are walking in the campus happily.

# Result: 
# It is very good. 

# Prompt: 
# A beautiful landscape with a river in the middle the left of the river is in the evening and in the winter with a big iceberg and a small village while some people are skating on the river and some people are skiing, 
# the right of the river is in the summer with a volcano in the morning and a small village while some people are playing.
# left top region, winter time with a a big iceberg, moon in the sky,
# right top region, summer time,a volcano eruption in the morning time,
# left bottom region, in the left side of the river,a small village ,people are skating on the river,
# right bottom region, summer time, a small viliage with green trees and flower

# Result: 
# The detail is not rich enough

# Prompt:
# cinematic photo Cinematic film still of an 18-year-old with very long blonde woman with sun-inspired macroscopic patterns. 
# Bright blue eyes. Highlight concentric circles, radiating sunbeams, and floral patterns in her clothing and background, 
# using a summery palette of warm yellows, oranges, and soft greens. sweet smiling. 
# Aim for a serene expression and warm, diffused lighting to emphasize the delicate sun-inspired textures.
# 8k UHD natural lighting, raw, rich, intricate details, key visual, 
# atmospheric lighting, professional, Shallow depth of field, vignette, high budget, cinemascope, vibrant, epic, gorgeous, 
# film grain, grainy, cinematic photorealistic,key visual, 35mm photograph,bokeh,
# 4k, colorful background, concept art, 8k, dramatic lighting, 
# high detail, hyper realistic, intricate, intricate sharp details, 
# octane render, smooth, studio lighting, trending on artstation. 
# film, highly detailed, slightly sweaty.

# Result:
# Very good