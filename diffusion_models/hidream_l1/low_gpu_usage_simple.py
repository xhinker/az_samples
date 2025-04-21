#%%
import torch
from diffusers import FlowMatchLCMScheduler, HiDreamImagePipeline
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

device = torch.device("cuda:0")

# repo_id             = "HiDream-ai/HiDream-I1-Dev"
# llama_repo          = "meta-llama/Llama-3.1-8B-Instruct"

model_type = "Dev"
repo_id             = f"/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/HiDream-ai/HiDream-I1-{model_type}_main"
llama_repo          = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated_main"

torch_dtype = torch.bfloat16

#%%
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    llama_repo,
)

text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch_dtype,
)

pipe = HiDreamImagePipeline.from_pretrained(
    repo_id,
    scheduler=None,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=None,
    vae=None,
    torch_dtype=torch_dtype,
)
pipe.enable_model_cpu_offload()

#%%
prompt              = "Ultra-realistic, high-quality photo of an anthropomorphic capybara with a tough, streetwise attitude, wearing a worn black leather jacket, dark sunglasses, and ripped jeans. The capybara is leaning casually against a gritty urban wall covered in vibrant graffiti. Behind it, in bold, dripping yellow spray paint, the word “HuggingFace” is scrawled in large street-art style letters. The scene is set in a dimly lit alleyway with moody lighting, scattered trash, and an edgy, rebellious vibe — like a character straight out of an underground comic book."
negative_prompt     = "bad quality, low quality"
with torch.no_grad():
    (
        prompt_embeds_t5,
        negative_prompt_embeds_t5,
        prompt_embeds_llama3,
        negative_prompt_embeds_llama3,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=False,
        device=device,
        dtype=torch_dtype,
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
    torch_dtype=torch_dtype,
)

pipe.transformer.enable_group_offload(
    onload_device=device,
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
    # low_cpu_mem_usage=True,
)
pipe.scheduler = FlowMatchLCMScheduler.from_config(pipe.scheduler.config, shift=6.0)
pipe.to(device)



#%%
image = pipe(
    prompt_embeds_t5=prompt_embeds_t5,
    prompt_embeds_llama3=prompt_embeds_llama3,
    negative_prompt_embeds_t5=negative_prompt_embeds_t5,
    negative_prompt_embeds_llama3=negative_prompt_embeds_llama3,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=28,
    generator=torch.Generator(device).manual_seed(43),
).images[0]

image.save("test.png")