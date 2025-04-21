#%%
import gc
import threading
import time

import psutil
import torch
from diffusers import HiDreamImagePipeline
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    T5EncoderModel,
    T5Tokenizer,
)


# get this scheduler from here: https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/schedulers/flash_flow_match.py
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler

# repo_id             = "HiDream-ai/HiDream-I1-Dev"
# transformer_repo_id = "HiDream-ai/HiDream-I1-Dev"
# llama_repo_id       = "meta-llama/Llama-3.1-8B-Instruct"

# model_type = "Full"
model_type = "Dev"
repo_id             = f"/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/HiDream-ai/HiDream-I1-{model_type}_main"
transformer_repo_id = f"/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/HiDream-ai/HiDream-I1-{model_type}_main"
llama_repo_id       = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated_main"

device              = torch.device("cuda")
torch_dtype         = torch.bfloat16
prompt              = "Ultra-realistic, high-quality photo of an anthropomorphic capybara with a tough, streetwise attitude, wearing a worn black leather jacket, dark sunglasses, and ripped jeans. The capybara is leaning casually against a gritty urban wall covered in vibrant graffiti. Behind it, in bold, dripping yellow spray paint, the word “HuggingFace” is scrawled in large street-art style letters. The scene is set in a dimly lit alleyway with moody lighting, scattered trash, and an edgy, rebellious vibe — like a character straight out of an underground comic book."

stop_monitoring = False

#%%
def log_memory_usage():
    """Logs RSS and VMS memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024**2)  # Resident Set Size in MB
    vms_mb = mem_info.vms / (1024**2)  # Virtual Memory Size in MB
    return rss_mb, vms_mb


def monitor_memory(interval, peak_memory_stats):
    """
    Monitors RSS and VMS memory usage at a given interval and updates peak values.
    Args:
        interval (float): Time interval between checks in seconds.
        peak_memory_stats (dict): A dictionary like {'rss': [0], 'vms': [0]}
                                  to store peak RSS and VMS values. The lists
                                  allow modification by reference from the thread.
    """
    global stop_monitoring  # Make sure to use the global flag if defined outside
    while not stop_monitoring:
        current_rss, current_vms = log_memory_usage()
        # Update peak RSS
        peak_memory_stats["rss"][0] = max(peak_memory_stats["rss"][0], current_rss)
        # Update peak VMS
        peak_memory_stats["vms"][0] = max(peak_memory_stats["vms"][0], current_vms)
        time.sleep(interval)


def flush(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(
        f"Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )
    print(f"Current CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def encode_prompt(
    prompt,
    pipeline_repo_id,
    llama_repo_id,
    do_classifier_free_guidance=False,
    device=device,
    dtype=torch_dtype,
):
    global stop_monitoring

    peak_memory_stats = {"rss": [0], "vms": [0]}
    initial_rss, initial_vms = log_memory_usage()
    print(f"Initial memory usage: RSS={initial_rss:.2f} MB, VMS={initial_vms:.2f} MB")

    stop_monitoring = False
    start_time = time.time()
    monitor_thread = threading.Thread(
        target=monitor_memory, args=(0.01, peak_memory_stats)
    )
    monitor_thread.start()

    prompt = [prompt] if isinstance(prompt, str) else prompt

    tokenizer_1 = CLIPTokenizer.from_pretrained(pipeline_repo_id, subfolder="tokenizer")
    # if this after the first run and offloaded the model, you can just move to `device`
    text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
        pipeline_repo_id, subfolder="text_encoder", torch_dtype=torch_dtype
    ).to(device)

    prompt_embeds = get_clip_prompt_embeds(prompt, tokenizer_1, text_encoder_1)
    prompt_embeds_1 = prompt_embeds.clone().detach()

    text_encoder_1.to("cpu")
    del prompt_embeds
    del tokenizer_1  # Don't delete if you have enough RAM
    del text_encoder_1  # Don't delete if you have enough RAM
    flush(device)

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        pipeline_repo_id, subfolder="tokenizer_2"
    )
    # if this after the first run and offloaded the model, you can just move to `device`
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pipeline_repo_id, subfolder="text_encoder_2", torch_dtype=torch_dtype
    ).to(device)

    prompt_embeds = get_clip_prompt_embeds(prompt, tokenizer_2, text_encoder_2)
    prompt_embeds_2 = prompt_embeds.clone().detach()

    text_encoder_2.to("cpu")
    del prompt_embeds
    del tokenizer_2  # Don't delete if you have enough RAM
    del text_encoder_2  # Don't delete if you have enough RAM
    flush(device)

    pooled_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    tokenizer_3 = T5Tokenizer.from_pretrained(
        pipeline_repo_id, subfolder="tokenizer_3", torch_dtype=torch_dtype
    )
    # if this after the first run and offloaded the model, you can just move to `device`
    text_encoder_3 = T5EncoderModel.from_pretrained(
        pipeline_repo_id, subfolder="text_encoder_3", torch_dtype=torch_dtype
    ).to(device)

    text_inputs = tokenizer_3(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask

    prompt_embeds = text_encoder_3(
        text_input_ids.to(device), attention_mask=attention_mask.to(device)
    )[0]
    t5_prompt_embeds = prompt_embeds.clone().detach()

    text_encoder_3.to("cpu")
    del prompt_embeds
    del text_inputs
    del tokenizer_3  # Don't delete if you have enough RAM
    del text_encoder_3  # Don't delete if you have enough RAM
    flush(device)

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_repo_id)
    tokenizer_4.pad_token = tokenizer_4.eos_token
    # if this after the first run and offloaded the model, you can just move to `device`
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llama_repo_id,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch_dtype,
    ).to(device)

    text_inputs = tokenizer_4(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask
    outputs = text_encoder_4(
        text_input_ids.to(device),
        attention_mask=attention_mask.to(device),
        output_hidden_states=True,
        output_attentions=True,
    )
    prompt_embeds = outputs.hidden_states[1:]
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    llama3_prompt_embeds = prompt_embeds.clone().detach()

    del prompt_embeds
    del outputs
    del text_inputs
    del tokenizer_4  # Don't delete if you have enough RAM
    del text_encoder_4  # Don't delete if you have enough RAM
    flush(device)

    prompt_embeds = [t5_prompt_embeds, llama3_prompt_embeds]

    end_time = time.time()
    stop_monitoring = True
    monitor_thread.join()

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(
        f"Peak memory usage: RSS={peak_memory_stats['rss'][0]:.2f} MB, VMS={peak_memory_stats['vms'][0]:.2f} MB"
    )
    final_rss, final_vms = log_memory_usage()
    print(f"Final memory usage: RSS={final_rss:.2f} MB, VMS={final_vms:.2f} MB")

    embeds = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": None,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": None,
    }

    return embeds


def get_clip_prompt_embeds(prompt, tokenizer, text_encoder):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


#%%
with torch.no_grad():
    embeddings = encode_prompt(
        prompt, repo_id, llama_repo_id, device=device, dtype=torch_dtype
    )

peak_memory_stats           = {"rss": [0], "vms": [0]}
initial_rss, initial_vms    = log_memory_usage()
print(f"Initial memory usage: RSS={initial_rss:.2f} MB, VMS={initial_vms:.2f} MB")

#%%
stop_monitoring = False
start_time = time.time()
monitor_thread = threading.Thread(target=monitor_memory, args=(0.01, peak_memory_stats))
monitor_thread.start()

#%%
from diffusers import HiDreamImageTransformer2DModel
transformer = HiDreamImageTransformer2DModel.from_pretrained(
    transformer_repo_id
    , subfolder       = "transformer"
    , torch_dtype     = torch.bfloat16
)

#%%
pipe = HiDreamImagePipeline.from_pretrained(
    repo_id,
    text_encoder=None,
    tokenizer=None,
    text_encoder_2=None,
    tokenizer_2=None,
    text_encoder_3=None,
    tokenizer_3=None,
    text_encoder_4=None,
    tokenizer_4=None,
    scheduler=None,
    #transformer=transformer,
    torch_dtype=torch_dtype,
)

#%%
pipe.transformer.enable_group_offload(
    onload_device=device,
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
    # low_cpu_mem_usage=True,  # use if you don't have enough RAM at the cost of inference speed
)

scheduler = FlashFlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000, shift=6.0, use_dynamic_shifting=False
)
pipe.scheduler = scheduler
pipe.to(device)

#%%
image = pipe(
    **embeddings,
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=28,
    generator=torch.Generator(device).manual_seed(42),
).images[0]

image.save("image.png")

end_time = time.time()
stop_monitoring = True
monitor_thread.join()

print(f"Time taken: {end_time - start_time:.2f} seconds")
print(
    f"Peak memory usage: RSS={peak_memory_stats['rss'][0]:.2f} MB, VMS={peak_memory_stats['vms'][0]:.2f} MB"
)
final_rss, final_vms = log_memory_usage()
print(f"Final memory usage: RSS={final_rss:.2f} MB, VMS={final_vms:.2f} MB")