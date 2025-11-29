"""Generate hierarchical dataset with merged classes (materials + objects)"""

import os
import yaml
import shutil
from pathlib import Path
from ..utils import load_config


def generate(config_path):
    """Generate dataset with merged classes: materials (e.g. 0-4) + objects (e.g. 5-18)"""

    # Load config
    config = load_config(config_path)

    source = Path(config["path_origin"])
    target = Path(config["path_target"])
    nc_m = config["nc_m"]

    print(f"Converting {source} -> {target}")

    # Handle target directory: delete if exists, then create
    if target.exists():
        print(f"Removing existing target directory: {target}")
        shutil.rmtree(target)
    
    print(f"Creating target directory: {target}")
    target.mkdir(parents=True, exist_ok=True)

    # Create output structure
    for split in ["train", "val", "test"]:
        (target / split / "images").mkdir(parents=True, exist_ok=True)
        (target / split / "labels").mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ["train", "val", "test"]:
        source_images = source / split / "images"
        source_labels = source / split / "labels"
        target_images = target / split / "images"
        target_labels = target / split / "labels"

        if not source_images.exists():
            continue

        count = 0
        for img_file in source_images.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                # Copy image
                shutil.copy2(img_file, target_images / img_file.name)

                # Process label
                label_file = source_labels / f"{img_file.stem}.txt"
                new_label_file = target_labels / f"{img_file.stem}.txt"

                if label_file.exists():
                    process_labels(label_file, new_label_file, nc_m)
                else:
                    new_label_file.touch()
                count += 1

        print(f"{split}: {count} images")

    # Create data.yaml
    all_names = config["names_m"] + config["names_o"]
    data_config = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(all_names),
        "names": all_names,
    }

    with open(target / "data.yaml", "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Done! Created {len(all_names)} classes")


def process_labels(source_file, target_file, offset):
    """Convert labels: new_class = original_class + offset"""
    with open(source_file, "r") as f:
        lines = f.readlines()

    with open(target_file, "w") as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                original_class = int(parts[0])
                new_class = original_class + offset
                new_line = f"{new_class} {' '.join(parts[1:5])}\n"
                f.write(new_line)
