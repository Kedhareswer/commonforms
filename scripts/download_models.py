from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "FFDNet-L.onnx": ("jbarrow/FFDNet-L-cpu", "FFDNet-L.onnx"),
    "FFDNet-S.onnx": ("jbarrow/FFDNet-S-cpu", "FFDNet-S.onnx"),
    "FFDetr.pth": ("jbarrow/FFDetr", "FFDetr.pth"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the CommonForms model weights into a local directory."
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODELS),
        default=None,
        help="Only download this model file",
    )
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    for filename, (repo_id, hub_filename) in MODELS.items():
        if args.model and filename != args.model:
            continue

        target = args.models_dir / filename
        if target.exists():
            print(f"skip {filename} (already present)")
            continue

        print(f"downloading {repo_id}/{hub_filename} ...")
        cached = hf_hub_download(repo_id=repo_id, filename=hub_filename)
        shutil.copyfile(cached, target)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
