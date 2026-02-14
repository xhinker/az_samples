#%%
import torch
from diffusers.utils import export_to_video
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, GGUFQuantizationConfig

ckpt_path = (
    "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q3_K_S.gguf"
)
transformer = LTXVideoTransformer3DModel.from_single_file(
    ckpt_path
    , quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    , torch_dtype         = torch.bfloat16
    , cache_dir           = "/home/andrewzhu/storage_14t_5/ai_models_all/hf_gguf_models"
)
#%%
pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video"
    , transformer   = transformer
    , generator     = torch.manual_seed(0)
    , torch_dtype   = torch.bfloat16
    , cache_dir     = "/home/andrewzhu/storage_14t_5/hf_download_cache"
)
pipe.enable_model_cpu_offload(gpu_id=0)

#%%
prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair. 
The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. 
The camera angle is a close-up, focused on the woman with brown hair's face. 
The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. 
The scene appears to be real-life footage
"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt                  = prompt
    , negative_prompt       = negative_prompt
    , width                 = 704
    , height                = 480
    , num_frames            = 161
    , num_inference_steps   = 50
).frames[0]
export_to_video(video, "output_gguf_ltx.mp4", fps=24)