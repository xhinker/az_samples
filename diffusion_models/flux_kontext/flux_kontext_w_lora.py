#%%
#%%
import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from torchao.quantization import quantize_, int8_weight_only
from azailib.image_tools import (
    resize_img
    , scale_img
    , concatenate_images_left_right
    , extend_mask_left
    , extract_object_on_white_background
    , extract_objects_using_xyxy_boxes
)

model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-Kontext-dev_main"

# load transformer model and quantize to int8
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder 	= "transformer"
    , torch_dtype 	= torch.bfloat16
)

pipe = FluxKontextPipeline.from_pretrained(
    model_path
    , transformer 	= transformer
    , torch_dtype 	= torch.bfloat16
)

#%%
lora_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/kontext_lora/flux-kontext-make-person-real-lora.safetensors"
pipe.load_lora_weights(lora_path, adapter_name="real")

#%%
quantize_(transformer, int8_weight_only())
pipe.enable_model_cpu_offload()

#%%
# # input_image_1 = load_image("https://shop.simon.com/cdn/shop/files/27683494fe4140139c23aa9633ae2604_1800x1800.jpg?v=1725423887")
# # input_image_2 = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1162.png")
# input_image_1 = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1279.png")
# input_image_2 = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1278.png")

# input_image = concatenate_images_left_right(input_image_1, input_image_2)
# display(input_image)

# # input_image_1 = resize_img(input_image_1, height=1200, width=864)
# # input_image_2 = resize_img(input_image_2, height=1200, width=864)

# # display(input_image_1)
# # display(input_image_2)
# w, h = input_image.size
# print(w,h)

#%%
# image_path = "/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1278.png"
image_path = "https://replicate.delivery/pbxt/NO4wpArlC8HvLWfxM9hQQqBrmG3rSjv2beS52dIYWc1f4q93/image.png"
input_image = load_image(image_path)
display(input_image)
w, h = input_image.size
print(w,h)

#%%
prompt = """make this person look realistic"""

lora_dict = {
    "real"              : 0.5 # 0.1 #0.125 #0.105
}
lora_name_list  = list(lora_dict.keys())
lora_value_list = list(lora_dict.values())
pipe.set_adapters(
    adapter_names       = lora_name_list
    ,adapter_weights    = lora_value_list
)

image = pipe(
	image           	    = input_image
	, prompt 			    = prompt
	, guidance_scale 	    = 2.5 #2.5
    , num_inference_steps   = 16
	, height 			    = h
    , width 			    = w
	, generator             = torch.Generator('cpu').manual_seed(4)
).images[0]
display(image)