#%%
import torch
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
# ***** Make sure accelerate is installed *****
try:
    import accelerate
except ImportError:
    raise ImportError("Please install accelerate: pip install accelerate")

#%%
# model_type = "full"
model_type = "dev"
# MODEL_PREFIX = "HiDream-ai"
# LLAMA_MODEL_NAME = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
MODEL_PREFIX        = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/azaneko"
LLAMA_MODEL_NAME    = "/home/andrewzhu/storage_14t_5/ai_models_all/llm_hf_models/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4_main"

# MODEL_PREFIX        = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/HiDream-ai"
# LLAMA_MODEL_NAME    = "/home/andrewzhu/storage_14t_5/ai_models_all/llm_hf_models/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4_main"

MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev-nf4_main",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full-nf4_main",
        "guidance_scale": 5.0,
        "num_inference_steps": 40,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}
# Resolution options (keep as is)
RESOLUTION_OPTIONS = [
    # ... (options remain the same)
]

# Load models (MODIFIED FUNCTION)
def load_models(model_type):
    config                          = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path   = config["path"]
    scheduler                       = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    max_memory_map = {
        0: "24GiB",  # Max memory for GPU 0
        1: "24GiB",  # Max memory for GPU 1
        "cpu": "0GiB" # Disallow model parameter allocation on CPU
    }    
    
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME
        #, use_fast  = False
        #, cache_dir = "models"
    )

    print("Loading Text Encoder (distributing across GPUs)...")
    # Load text encoder with automatic device mapping
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME
        , output_hidden_states    = True
        , output_attentions       = True
        , return_dict_in_generate = True
        , torch_dtype             = torch.bfloat16
        , cache_dir               = "models"
        # ***** Use accelerate's automatic device map *****
        , device_map              = "auto"
        , max_memory              = max_memory_map 
        # ***** REMOVE .to("cuda") *****
    )
    print("Text Encoder loaded.")

    print("Loading Transformer (distributing across GPUs)...")
    # Load transformer with automatic device mapping
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder       = "transformer",
        torch_dtype     = torch.bfloat16,
        cache_dir       = "models",
        # ***** Use accelerate's automatic device map *****
        device_map      = "auto",
        max_memory      = max_memory_map 
        # ***** REMOVE .to("cuda") *****
    )
    print("Transformer loaded.")


    print("Loading Pipeline...")
    # Load the pipeline, passing the already distributed components
    # Do NOT use .to("cuda") on the pipeline itself, as its components are already mapped
    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path
        , scheduler       = scheduler
        , tokenizer_4     = tokenizer_4
        , text_encoder_4  = text_encoder_4 # Pass the distributed text encoder
        , torch_dtype     = torch.bfloat16
        #, cache_dir       = "models"
        # ***** REMOVE .to("cuda", torch.bfloat16) *****
    )
    # Assign the distributed transformer
    pipe.transformer = transformer
    print("Pipeline loaded.")

    # Note: The pipeline itself doesn't need .to(device) because its main
    # heavy components (text_encoder, transformer) are already distributed
    # via device_map="auto". Accelerate hooks handle moving data correctly.

    # --- FIX START ---
    # Explicitly move the VAE component to the GPU
    # Use the device of one of the already distributed components as the target
    
    # target_device = pipe.transformer.device # Or text_encoder_4.device, or just "cuda:0"
    # print(f"Moving VAE to device: {target_device}...")
    # pipe.vae.to(target_device)
    # print("VAE moved.")
    # --- FIX END ---
    
    #pipe.enable_sequential_cpu_offload()
    pipe.enable_model_cpu_offload()
    
    return pipe, config

# Parse resolution function (keep as is)
def parse_resolution(resolution_str):
    # ... (function remains the same)
    pass

# Generate image function (keep as is - Generator targeting "cuda" is fine,
# it usually defaults to cuda:0 for generation init, accelerate handles the rest)
def generate_image(pipe, model_type, prompt, resolution, seed):
    config              = MODEL_CONFIGS[model_type]
    guidance_scale      = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    height, width       = resolution #parse_resolution(resolution)

    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()

    # Generator on default cuda device is fine
    generator = torch.Generator("cuda").manual_seed(seed)

    print("REached this point")

    print(f"Generating image with seed: {seed}")
    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator
    ).images

    return images[0], seed

#%%
# --- Main Execution ---
print("Loading default model (full)...")
pipe, _ = load_models(model_type)
print("Model loaded successfully!")

#%%
national = "korean"
# prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"."
# prompt = """
# Fashion portrait photo of thin young woman, 25yo, 
# with light green almond-shaped eyes and narrow lips with an arrogant attitude. 
# She has tousled auburn emo scene hair and emo makeup featuring shades of white and mascara with glossy lipstick, 
# wearing sleeveless faux leather graphic tee in combination with a beaded necklace. 
# Set on an european castle visit with authentic medieval walls, 
# towers and stone arches with a cobbled path leading through, 
# on a sunny rainy sunrise in summer with a shy smile. With artificial split lighting, 
# shallow depth of field, dynamic angle
# """

# prompt = """
# Exquisite and alluring woman exuding elegance and confidence, 
# dressed in a sophisticated and stylish outfit that complements her figure. Soft, 
# flattering lighting enhances her features, creating a captivating and sensual ambiance. 
# Her pose is graceful and enticing, adding to the overall allure. 
# The background is simple yet tasteful, allowing the focus to remain on her beauty.
# perfect body
# """

# prompt = f"""
# 8k, best quality, masterpiece, realistic, photo-realistic, ultra detailed, 
# sharp focus, raw photo, natural lighting, Lumix GH5, Voigtlnder Nokton 50mm f1.1,
# glamour photography of young adorable petite toned shy ({national}) virgin bride, 
# she is stunning beautiful and amazingly good looking, 
# (wear suit outside, unbutton, naked between suit collar:1.5), 
# (no pants wear:1.4), no shirt wear, 
# (wear Pantyhose:1.5), 
# curly long hair, french braid, 
# black eyes, dark raven hair,
# (perfect round breasts are completely exposed:1.5),
# (detailed areola, detailed nipples),
# (show perky breasts, show nipples:1.5),
# (pussy and vaginal is vaguely exposed:1.2),
# standing by a river, trees and flower around,
# (the most sexy body:1.4)
# (perfect sexy body shape:1.5)
# """

prompt = f"""
glamour photography, full body photo of young adorable petite toned shy korean virgin bride, 
she is stunning beautiful and amazingly good looking, 
wear suit outside, wear lace inside, unbutton, naked between suit collar, 
no pants wear, no shirt wear, wear Pantyhose, curly long hair, french braid, 
black eyes, dark raven hair,perfect round breasts are completely exposed,
detailed areola, detailed nipples,show perky breasts, show nipples,
stand on sunset beach, buttocks, perfect body, wear leather shoe perfect sexy body shape
"""

resolution = 1024, 1024
seed = -1 # Use -1 for random seed

image, used_seed = generate_image(pipe, model_type, prompt, resolution, seed)

print(f"Image generated with seed: {used_seed}")
output_filename = f"output_seed_{used_seed}.png"
image.save(output_filename)
print(f"Image saved to {output_filename}")

# image.show(title=f"HiDream Result (Seed: {used_seed})") 
display(image)