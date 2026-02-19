import os
import multiprocessing as mp

# Env vars MUST be set before any torch/vllm import
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.LLM(
        model="/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-asr/models/tts_hf_models/Qwen/Qwen3-ASR-1.7B_main",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128,
        max_new_tokens=4096,
        enforce_eager=True,
        forced_aligner="/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/voice_models/qwen3-asr/models/tts_hf_models/Qwen/Qwen3-ForcedAligner-0.6B_main",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda",
        ),
    )

    results = model.transcribe(
        audio=[
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        ],
        language=["Chinese", "English"],
        return_time_stamps=True,
    )

    for r in results:
        print(r.language, r.text, r.time_stamps[0])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()