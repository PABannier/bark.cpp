import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


BARK_SMALL_REPO_ID = "suno/bark-small"
BARK_REPO_ID = "suno/bark"

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--models", nargs="+", default=["bark-small", "bark"], required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        if model == "bark-small":
            print("     -> Downloading Bark-Small...")
            bark_small_out_dir = out_dir / "bark-small"
            bark_small_out_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                BARK_SMALL_REPO_ID,
                local_dir=bark_small_out_dir,
                allow_patterns=["*.bin", "*.json", "*.txt"]
            )
        elif model == "bark":
            print("     -> Downloading Bark...")
            bark_out_dir = out_dir / "bark"
            bark_out_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                BARK_REPO_ID,
                local_dir=bark_out_dir,
                allow_patterns=["*.bin", "*.json", "*.txt"]
            )

    print("Done.")
