from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 5-class MIT Indoor Scenes subset.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to MIT Indoor Scenes Images directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output subset directory.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["classroom", "computerroom", "library", "corridor", "office"],
    )
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    for class_name in args.classes:
        src = source_dir / class_name
        if not src.exists():
            raise FileNotFoundError(f"Class folder not found: {src}")

        dst = output_dir / class_name
        dst.mkdir(parents=True, exist_ok=True)

        images = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        images = sorted(images)
        if args.max_per_class is not None:
            rng.shuffle(images)
            images = images[: args.max_per_class]

        for p in images:
            shutil.copy2(p, dst / p.name)

        print(f"{class_name}: copied {len(images)} images -> {dst}")

    print(f"Done. Subset saved to: {output_dir}")


if __name__ == "__main__":
    main()
