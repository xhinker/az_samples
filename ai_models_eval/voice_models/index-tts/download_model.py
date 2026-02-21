from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import snapshot_download


def download_hf_model(model_type: str, model_id: str, hf_branch: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    local_dir = script_dir / "models" / model_type / f"{model_id}_{hf_branch}"
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        revision=hf_branch,
        local_dir=str(local_dir),
    )

    return local_dir


def parse_args() -> tuple[str, str, str]:
    parser = ArgumentParser(
        description="Download a Hugging Face model into moss-tts peer models folder."
    )
    parser.add_argument("model_type", help="Model type subfolder name under models")
    parser.add_argument(
        "model_id",
        help="Hugging Face model id, e.g. OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    )
    parser.add_argument("hf_branch", help="Hugging Face branch/revision, e.g. main")

    args = parser.parse_args()
    return args.model_type, args.model_id, args.hf_branch


if __name__ == "__main__":
    model_type_arg, model_id_arg, hf_branch_arg = parse_args()
    output_dir = download_hf_model(model_type_arg, model_id_arg, hf_branch_arg)
    print(f"Downloaded to: {output_dir}")


# python3 ./download_model.py tts_hf_models IndexTeam/IndexTTS-2 main