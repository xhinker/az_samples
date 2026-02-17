import os
from pathlib import Path
import wave
import torch
from transformers import AutoModel, AutoProcessor, GenerationConfig
# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32
        
def initial_config(tokenizer, model_name_or_path):
    generation_config = DelayGenerationConfig.from_pretrained(model_name_or_path)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = 151653
    generation_config.max_new_tokens = 1000000
    generation_config.temperature = 1.0
    generation_config.repetition_penalty = 1.1
    generation_config.use_cache = True
    generation_config.do_sample = False
    return generation_config


def save_wav_pcm16(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    if audio.dim() == 2:
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)
        elif audio.shape[1] == 1:
            audio = audio.squeeze(1)
        else:
            raise ValueError(f"Expected mono audio, got shape={tuple(audio.shape)}")

    audio = audio.detach().to(torch.float32).cpu().clamp(-1.0, 1.0)
    pcm16_bytes = (audio * 32767.0).round().to(torch.int16).numpy().tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16_bytes)


def load_wav_tensor(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        data = torch.frombuffer(memoryview(raw), dtype=torch.uint8).clone()
        audio = data.to(torch.float32).sub(128.0).div(128.0)
    elif sample_width == 2:
        data = torch.frombuffer(memoryview(raw), dtype=torch.int16).clone()
        audio = data.to(torch.float32).div(32768.0)
    elif sample_width == 3:
        # 24-bit little-endian PCM -> signed int32, then normalize to [-1, 1).
        data_u8 = torch.frombuffer(memoryview(raw), dtype=torch.uint8).clone().view(-1, 3)
        audio_i32 = (
            data_u8[:, 0].to(torch.int32)
            | (data_u8[:, 1].to(torch.int32) << 8)
            | (data_u8[:, 2].to(torch.int32) << 16)
        )
        audio_i32 = torch.where(
            audio_i32 >= 0x800000,
            audio_i32 - 0x1000000,
            audio_i32,
        )
        audio = audio_i32.to(torch.float32).div(8388608.0)
    elif sample_width == 4:
        data = torch.frombuffer(memoryview(raw), dtype=torch.int32).clone()
        audio = data.to(torch.float32).div(2147483648.0)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    audio = audio.view(-1, num_channels).transpose(0, 1).contiguous()
    return audio, int(sample_rate)



# pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS"
pretrained_model_name_or_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/moss-tts/models/tts_hf_models/OpenMOSS-Team/MOSS-TTS-Local-Transformer_main"
codec_model_name_or_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/moss-tts/models/tts_hf_models/OpenMOSS-Team/MOSS-Audio-Tokenizer_main"

if not Path(codec_model_name_or_path).exists():
    raise FileNotFoundError(
        "Missing local codec model at "
        f"{codec_model_name_or_path}. "
        "Download it first with:\n"
        "python ./download_model.py tts_hf_models OpenMOSS-Team/MOSS-Audio-Tokenizer main"
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    codec_path=codec_model_name_or_path,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

assets_audio_dir = Path(__file__).resolve().parent / "repos" / "MOSS-TTS" / "assets" / "audio"
ref_wav_1 = assets_audio_dir / "reference_zh.wav"
ref_wav_2 = assets_audio_dir / "reference_zh_1.wav"
if not ref_wav_1.exists() or not ref_wav_2.exists():
    raise FileNotFoundError(
        "Local reference wav files not found. Expected:\n"
        f"- {ref_wav_1}\n"
        f"- {ref_wav_2}"
    )

ref_wav_tensor_1, ref_sr_1 = load_wav_tensor(ref_wav_1)
ref_wav_tensor_2, ref_sr_2 = load_wav_tensor(ref_wav_2)
ref_audio_codes_1 = processor.encode_audios_from_wav([ref_wav_tensor_1], ref_sr_1)[0]
ref_audio_codes_2 = processor.encode_audios_from_wav([ref_wav_tensor_2], ref_sr_2)[0]

text_1 = """亲爱的你，
你好呀。

今天，我想用最认真、最温柔的声音，对你说一些重要的话。
这些话，像一颗小小的星星，希望能在你的心里慢慢发光。

首先，我想祝你——
每天都能平平安安、快快乐乐。

希望你早上醒来的时候，
窗外有光，屋子里很安静，
你的心是轻轻的，没有着急，也没有害怕。
"""
text_2 = """
We stand on the threshold of the AI era.
Artificial intelligence is no longer just a concept in laboratories, but is entering every industry, every creative endeavor, and every decision. It has learned to see, hear, speak, and think, and is beginning to become an extension of human capabilities. AI is not about replacing humans, but about amplifying human creativity, making knowledge more equitable, more efficient, and allowing imagination to reach further. A new era, jointly shaped by humans and intelligent systems, has arrived.
"""
text_3 = "nin2 hao3，qing3 wen4 nin2 lai2 zi4 na3 zuo4 cheng2 shi4？"
text_4 = "nin2 hao3，qing4 wen3 nin2 lai2 zi4 na4 zuo3 cheng4 shi3？"
text_5 = "您好，请问您来自哪 zuo4 cheng2 shi4？"
text_6 = "/həloʊ, meɪ aɪ æsk wɪtʃ sɪti juː ɑːr frʌm?/"

conversations = [
    # Direct TTS (no reference)
    [
        processor.build_user_message(text=text_1)
    ],
    [
        processor.build_user_message(text=text_2)
    ]
    # ,
    # # Pinyin or IPA input
    # [
    #     processor.build_user_message(text=text_3)
    # ],
    # [
    #     processor.build_user_message(text=text_4)
    # ],
    # [
    #     processor.build_user_message(text=text_5)
    # ],
    # [
    #     processor.build_user_message(text=text_6)
    # ],
    # # Voice cloning (with reference)
    # [
    #     processor.build_user_message(text=text_1, reference=[ref_audio_codes_1])
    # ],
    # [
    #     processor.build_user_message(text=text_2, reference=[ref_audio_codes_2])
    # ],
]



model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=dtype,
    local_files_only=True,
).to(device)
model.eval()

generation_config = initial_config(processor.tokenizer, pretrained_model_name_or_path)
generation_config.n_vq_for_inference = model.channels - 1
generation_config.do_samples = [True] * model.channels
generation_config.layers = [
    {
        "repetition_penalty": 1.0, 
        "temperature": 1.5, 
        "top_p": 1.0, 
        "top_k": 50
    }
] + [ 
    {
        "repetition_penalty": 1.1, 
        "temperature": 1.0, 
        "top_p": 0.95,
        "top_k": 50
    }
] * (model.channels - 1) 

batch_size = 1

messages = []
save_dir = Path(f"inference_root_moss_tts_local_transformer_generation")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

        for message in processor.decode(outputs):
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            save_wav_pcm16(out_path, audio, processor.model_config.sampling_rate)
