#%%
from diffusers import FluxControlPipeline
from image_gen_aux import DepthPreprocessor
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
import torch

#%%
model_path      = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
lora_path_1     = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/AsirAsianPhotographyflux.safetensors"
lora_path_2     = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_lora/nwsj_flux0924.safetensors"

control_pipe = FluxControlPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
control_pipe.load_lora_weights(lora_path_1, adapter_name="asian")
control_pipe.load_lora_weights(lora_path_2, adapter_name="nwsj")
# control_pipe.load_lora_weights(
#     hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), adapter_name="hyper-sd"
# )
control_pipe.set_adapters(["asian", "nwsj"], adapter_weights=[0.85, 0.125])
control_pipe.enable_model_cpu_offload()

#%%
prompt = """
a photo of a 20yo blonde woman with large breasts, ridingsexscene, nipples, nude, open mouth, closed eyes, long hair, navel, realistic, from below, arms behind head, ceiling corner
"""
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = control_pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("output.png")