'''
use diffusers==0.30.2
Try downgrade your diffusers==0.30.2 and transformers==4.42.0 (optional)

Seems not very good so far, seems under trained. 
'''
#%%
import torch
from diffusers.utils import load_image, check_min_version
from flux_lib.controlnet_flux import FluxControlNetModel
from flux_lib.transformer_flux import FluxTransformer2DModel
from flux_lib.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from torchao.quantization import quantize_, int8_weight_only
check_min_version("0.30.2")

checkpoint_model_path   = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
controlnet_model_path   = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta_main"
device                  = "cuda:0"

#%%
controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model_path
    , torch_dtype=torch.bfloat16
)

transformer = FluxTransformer2DModel.from_pretrained(
    checkpoint_model_path
    , subfolder     = 'transformer'
    , torch_dtype   = torch.bfloat16
)
quantize_(
    transformer
    , int8_weight_only()
    , device = device
)

pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    checkpoint_model_path
    , controlnet    = controlnet
    , transformer   = transformer
    , torch_dtype   = torch.bfloat16
)
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

pipe.enable_model_cpu_offload()

#%%
# image_path     = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_path    = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
# prompt      = "A pure white cat sitting on a bench"

image_path  = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/images/jeans.png'
mask_path   = '/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/mask.png'
prompt      = """\
her upbody wear pink Plaid shirt
"""

# Set image path , mask path and prompt
# image_path  = 'https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket.png'
# mask_path   = 'https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket_mask.jpeg'
# prompt      = 'a person wearing a white shoe, carrying a white bucket with text "FLUX" on it'

# Load image and mask
image = load_image(image_path).convert("RGB")#.resize(size)
size = image.size
display(image)
print(size)
mask = load_image(mask_path).convert("RGB").resize(size)
display(mask)

#%%
# Inpaint
result = pipe(
    prompt                          = prompt
    , height                        = size[1]
    , width                         = size[0]
    , control_image                 = image
    , control_mask                  = mask
    , num_inference_steps           = 25
    , generator                     = torch.Generator(device="cuda").manual_seed(123)
    , controlnet_conditioning_scale = 0.75
    , guidance_scale                = 3.5
    , negative_prompt               = ""
    , true_guidance_scale           = 1.2 #1.0 # default: 3.5 for alpha and 1.0 for beta
).images[0]

result.save('output_images/flux_inpaint.png')
display(result)
print("Successfully inpaint image")
