#%%
import torch
from diffusers.utils import export_to_video
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, LTXLatentUpsamplePipeline, LTXConditionPipeline
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

transformer = LTXVideoTransformer3DModel.from_single_file(
    # "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/ltxvideo/ltxv-2b-0.9.8-distilled-fp8.safetensors"
    "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/ltxvideo/ltxv-13b-0.9.8-dev-fp8.safetensors"
    # , quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    , torch_dtype         = torch.bfloat16
    # , cache_dir           = "/home/andrewzhu/storage_14t_5/ai_models_all/hf_gguf_models"
)

pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.8-13B-distilled"
    , transformer   = transformer
    , generator     = torch.manual_seed(0)
    , torch_dtype   = torch.bfloat16
    , cache_dir     = "/home/andrewzhu/storage_14t_5/hf_download_cache"
)

pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    "linoyts/LTX-Video-spatial-upscaler-0.9.8"
    , vae           = pipe.vae
    , torch_dtype   = torch.bfloat16
    , cache_dir     = "/home/andrewzhu/storage_14t_5/hf_download_cache"
)

#%%
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
quantize_(transformer, int8_weight_only())
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()
# pipe.to("cuda")
# pipe_upsample.enable_model_cpu_offload()

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

#%%
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png")
# image = load_image("/home/andrewzhu/storage_8t_4/sd_input_output/202506/flux/img2img_output_seed_1200.png")
display(image)
video       = load_video(export_to_video([image])) # compress the image using video compression as the model was trained on videos
condition1  = LTXVideoCondition(video=video, frame_index=0)

prompt                          = "A cute little penguin takes out a book and starts reading it"
# prompt = """The girl is a fashion model, she moves body and hands gracefully to show her cloth, 
# fashion show style, with clear movements,sweet smile, full of charm.
# """
negative_prompt                 = "worst quality, inconsistent motion, blurry, jittery, distorted"
expected_height, expected_width = 832,480
downscale_factor                = 2 / 3
num_frames                      = 120 #96

# Part 1. Generate video at smaller resolution
downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
latents = pipe(
    conditions          = [condition1],
    prompt              = prompt,
    negative_prompt     = negative_prompt,
    width               = downscaled_width,
    height              = downscaled_height,
    num_frames          = num_frames,
    num_inference_steps = 30,
    generator           = torch.Generator().manual_seed(0),
    output_type         = "latent",
).frames

#%%
# Part 2. Upscale generated video using latent upsampler with fewer inference steps
# The available latent upsampler upscales the height/width by 2x
pipe_upsample.to("cuda:0")
upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
upscaled_latents = pipe_upsample(
    latents         =latents,
    output_type     ="latent"
).frames
pipe_upsample.to("cpu")

#%%
# Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
video = pipe(
    conditions              = [condition1],
    prompt                  = prompt,
    negative_prompt         = negative_prompt,
    width                   = upscaled_width,
    height                  = upscaled_height,
    num_frames              = num_frames,
    denoise_strength        = 0.4,  # Effectively, 4 inference steps out of 10
    num_inference_steps     = 10,
    latents                 = upscaled_latents,
    decode_timestep         = 0.05,
    image_cond_noise_scale  = 0.025,
    generator               = torch.Generator().manual_seed(0),
    output_type             = "pil",
).frames[0]

# Part 4. Downscale the video to the expected resolution
video = [frame.resize((expected_width, expected_height)) for frame in video]

export_to_video(video, "output.mp4", fps=24)