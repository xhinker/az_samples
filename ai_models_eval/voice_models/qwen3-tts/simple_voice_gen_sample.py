#%%
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model_id = "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice_main"
# model_id = "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice_main"

model = Qwen3TTSModel.from_pretrained(
    model_id
    , device_map            = "cuda:0"
    , dtype                 = torch.bfloat16
    , attn_implementation   = "flash_attention_2"
)

print(model.model.dtype)                 # should be torch.bfloat16
print(model.model.config.torch_dtype)    # should be torch.bfloat16

#%% Single inference - English
wavs, sr = model.generate_custom_voice(
    text        = "Actually, I've really discovered that I'm someone who's particularly good at observing other people's emotions.",
    language    = "Auto", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker     = "Vivian",
    instruct    = "Speak in a particularly gentle tone", # Omit if not needed.
)
sf.write("output_custom_voice_en.wav", wavs[0], sr)

#%%
# single inference - Chinese
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别性感的语气说", # Omit if not needed.
    # instruct="用讽刺的语气说", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)

#%%
# batch inference
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."]
)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)


#%% Voice design ---------------------------------------------------------------
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
model_id = "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign_main"
model = Qwen3TTSModel.from_pretrained(
    model_id,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

#%%
# single inference
wavs, sr = model.generate_voice_design(
    # text        ="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    text        ="The difference between moral imperative and moral obligation lies in their scope, origin, and nuance—though they are closely related and sometimes used interchangeably.",
    language    = "Auto", #"Chinese",
    # instruct    ="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
    # instruct    ="用特别温柔性感的语气说",
    instruct    ="a deep male voice with a slight rasp and a British accent",
)
sf.write("output_voice_design_en.wav", wavs[0], sr)


#%% with audio reference -------------------------------------------------------
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained(
    "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main"
    , device_map            = "cuda:0"
    , dtype                 = torch.bfloat16
    , attn_implementation   = "flash_attention_2"
)

#%% generate voice embedding
ref_audio = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-tts/outputs/output_custom_voice_en.wav"
ref_text  = "Actually, I've really discovered that I'm someone who's particularly good at observing other people's emotions."

ref_audio_embedding = model.create_voice_clone_prompt(
    ref_audio               = ref_audio
    , ref_text              = ref_text
    , x_vector_only_mode    = False
)

#%%
wavs, sr = model.generate_voice_clone(
    text                    = ["I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!"]
    , language              = ["English"]
    , voice_clone_prompt    = ref_audio_embedding
)
sf.write("outputs/output_voice_clone.wav", wavs[0], sr)

#%% audio streaming test in VS Code interactive cell
import numpy as np
import re
import time
import torch
from IPython.display import Audio, display
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "models/tts_hf_models/Qwen/Qwen3-TTS-12Hz-1.7B-Base_main"
    , device_map            = "cuda:0"
    , dtype                 = torch.bfloat16
    , attn_implementation   = "flash_attention_2"
)

#%%
ref_audio = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-tts/outputs/output_custom_voice_en.wav"
ref_text  = "Actually, I've really discovered that I'm someone who's particularly good at observing other people's emotions."
ref_audio_embedding = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False
)

#%%
text_to_speak = """
This is a streaming test using the cloned voice. I should hear the audio almost immediately.
Actually, I've really discovered that I'm someone who's particularly good at observing other people's emotions.
"""

# This qwen_tts wrapper does not expose true audio-token streaming in Python.
# Practical workaround: split text into short chunks and generate each chunk on demand.
TARGET_CHUNK_SECONDS = 2.0
WORDS_PER_SECOND = 2.5
WORDS_PER_CHUNK = max(1, int(TARGET_CHUNK_SECONDS * WORDS_PER_SECOND))


def split_text_for_live_gen(text: str, words_per_chunk: int) -> list[str]:
    text = " ".join(text.strip().split())
    if not text:
        return []
    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", text) if s]
    chunks = []
    for sent in sentences:
        words = sent.split()
        while len(words) > words_per_chunk:
            chunks.append(" ".join(words[:words_per_chunk]))
            words = words[words_per_chunk:]
        if words:
            chunks.append(" ".join(words))
    return chunks


text_chunks = split_text_for_live_gen(text_to_speak, words_per_chunk=WORDS_PER_CHUNK)
if not text_chunks:
    raise RuntimeError("No text to generate.")

print(f"Planned {len(text_chunks)} generation chunks (target ~{TARGET_CHUNK_SECONDS:.1f}s each).")
for idx, text_chunk in enumerate(text_chunks, start=1):
    t0 = time.perf_counter()
    wavs, sample_rate = model.generate_voice_clone(
        text=text_chunk,
        voice_clone_prompt=ref_audio_embedding,
        language="English",
        non_streaming_mode=False,
    )
    gen_s = time.perf_counter() - t0
    audio_float = np.asarray(wavs[0], dtype=np.float32).flatten()
    if audio_float.size == 0:
        print(f"Chunk {idx}: empty output, skipped.")
        continue
    audio_s = audio_float.size / float(sample_rate)
    print(f"Chunk {idx}/{len(text_chunks)} | gen={gen_s:.2f}s | audio={audio_s:.2f}s | text='{text_chunk}'")
    display(Audio(audio_float, rate=sample_rate, autoplay=True))
    time.sleep(max(0.0, audio_s))
