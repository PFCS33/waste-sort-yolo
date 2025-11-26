import os
import shutil
import random
from collections import defaultdict


def merge_all(config, split=[0.8, 0.1, 0.1]):
    """
    Merge all transformed datasets and split into train/val/test with balanced class distribution

    Args:
        config: Configuration dictionary
        split: Split ratios for [train, val, test], default [0.8, 0.1, 0.1]
    """
    root_dir = config["global"]["root_dir"]
    custom_name = config["global"]["custom_name"]

    print(f"Starting merge_all: merging datasets into {custom_name}")

    # 1: find all transformed datasets
    transformed_datasets = collect_transformed_datasets(config)
    if not transformed_datasets:
        print("No transformed datasets found!")
        return

    # 2: setup merge dir
    merge_dir = os.path.join(root_dir, custom_name)
    origin_dir = os.path.join(merge_dir, "origin")
    origin_images = os.path.join(origin_dir, "images")
    origin_labels = os.path.join(origin_dir, "labels")
    # clean up old dir
    if os.path.exists(merge_dir):
        print(f"Removing existing merge directory: {merge_dir}")
        shutil.rmtree(merge_dir)
    os.makedirs(origin_images, exist_ok=True)
    os.makedirs(origin_labels, exist_ok=True)

    # 3: merge all datasets
    merge_datasets(transformed_datasets, origin_images, origin_labels)

    # 4: analyze class distribution
    class_distribution = analyze_class_distribution(origin_labels)

    # 5: create balanced splits
    balance_splits(merge_dir, origin_dir, class_distribution, split)

    print(f"\n✓ Merge complete! Results saved in {merge_dir}")


def collect_transformed_datasets(config):
    """collect image/label dir for each dataset"""
    root_dir = config["global"]["root_dir"]
    datasets = []

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        if not dataset_name:
            continue

        dataset_root = os.path.join(root_dir, dataset_name)
        transformed_dir = os.path.join(dataset_root, "transformed")

        if os.path.exists(transformed_dir):
            images_dir = os.path.join(transformed_dir, "images")
            labels_dir = os.path.join(transformed_dir, "labels")

            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                print(f"Found dataset: {dataset_name}")
                datasets.append(
                    {
                        "name": dataset_name,
                        "images_dir": images_dir,
                        "labels_dir": labels_dir,
                    }
                )

    print(f"Total transformed datasets found: {len(datasets)}")
    return datasets


def merge_datasets(datasets, target_images, target_labels):
    """merge all datasets into target directories with continuous indexing"""
    current_index = 0
    total_merged = 0

    for dataset in datasets:
        print(f"Merging {dataset['name']}...")

        images_dir = dataset["images_dir"]
        labels_dir = dataset["labels_dir"]

        # Get all image files and sort for consistent processing
        image_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_files.sort()

        print(f"  Processing {len(image_files)} images")

        for image_file in image_files:
            # Copy image with new index
            old_image_path = os.path.join(images_dir, image_file)
            file_ext = os.path.splitext(image_file)[1]
            new_image_name = f"{current_index}{file_ext}"
            new_image_path = os.path.join(target_images, new_image_name)
            shutil.copy2(old_image_path, new_image_path)

            # Copy corresponding label
            label_file = os.path.splitext(image_file)[0] + ".txt"
            old_label_path = os.path.join(labels_dir, label_file)
            new_label_name = f"{current_index}.txt"
            new_label_path = os.path.join(target_labels, new_label_name)

            if os.path.exists(old_label_path):
                shutil.copy2(old_label_path, new_label_path)
            else:
                print(f"Warning: Label not found for {image_file}")

            current_index += 1

        merged_count = len(image_files)
        total_merged += merged_count
        print(f"  ✓ Merged {merged_count} samples from {dataset['name']}")

    print(f"Total samples merged: {total_merged}")
    return total_merged


def analyze_class_distribution(labels_dir):
    """calculate class distribution in merged dataset, return rougth class assignment of each image"""
    class_counts = defaultdict(int)
    sample_classes = defaultdict(list)  # track which samples belong to each class

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        sample_id = os.path.splitext(label_file)[0]

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            sample_classes_set = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    cls_id = int(float(parts[0]))
                    sample_classes_set.add(cls_id)

            # for multi-class samples, assign to the first class found
            if sample_classes_set:
                primary_class = min(sample_classes_set)
                class_counts[primary_class] += 1
                sample_classes[primary_class].append(sample_id)

        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue

    print("Class distribution:")
    total_samples = sum(class_counts.values())
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  Class {cls_id}: {count} samples ({percentage:.1f}%)")

    return dict(sample_classes)


def balance_splits(merge_dir, origin_dir, sample_classes, split_ratios):
    """create balanced train/val/test splits"""
    train_ratio, val_ratio, _ = split_ratios

    # clean up old split directories
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(merge_dir, split_name)
        if os.path.exists(split_dir):
            print(f"Removing existing {split_name} directory: {split_dir}")
            shutil.rmtree(split_dir)

    # create split directories
    for split_name in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            split_path = os.path.join(merge_dir, split_name, subdir)
            os.makedirs(split_path, exist_ok=True)

    origin_images = os.path.join(origin_dir, "images")
    origin_labels = os.path.join(origin_dir, "labels")

    split_assignments = {"train": [], "val": [], "test": []}

    # for each class, split samples proportionally
    for cls_id, samples in sample_classes.items():
        random.shuffle(samples)
        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val :]

        split_assignments["train"].extend(train_samples)
        split_assignments["val"].extend(val_samples)
        split_assignments["test"].extend(test_samples)


    # move files to appropriate split dir
    for split_name, sample_ids in split_assignments.items():
        print(f"\nMoving {len(sample_ids)} samples to {split_name}...")

        target_images = os.path.join(merge_dir, split_name, "images")
        target_labels = os.path.join(merge_dir, split_name, "labels")

        for sample_id in sample_ids:
            # move image
            for ext in [".jpg", ".jpeg", ".png"]:
                origin_img_path = os.path.join(origin_images, f"{sample_id}{ext}")
                if os.path.exists(origin_img_path):
                    target_img_path = os.path.join(target_images, f"{sample_id}{ext}")
                    shutil.move(origin_img_path, target_img_path)
                    break

            # move label
            origin_label_path = os.path.join(origin_labels, f"{sample_id}.txt")
            if os.path.exists(origin_label_path):
                target_label_path = os.path.join(target_labels, f"{sample_id}.txt")
                shutil.move(origin_label_path, target_label_path)

    # Clean up origin directory
    if os.path.exists(origin_dir):
        shutil.rmtree(origin_dir)
        print("Cleaned up temporary origin directory")

    # Print class distribution in each split
    print("\nClass distribution in each split:")
    for split_name in ["train", "val", "test"]:
        split_labels = os.path.join(merge_dir, split_name, "labels")
        split_class_counts = defaultdict(int)
        
        label_files = [f for f in os.listdir(split_labels) if f.endswith(".txt")]
        total_samples = len(label_files)
        
        # Count classes in this split
        for label_file in label_files:
            label_path = os.path.join(split_labels, label_file)
            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()
                if lines:
                    first_line = lines[0].strip().split()
                    if len(first_line) >= 1:
                        cls_id = int(float(first_line[0]))
                        split_class_counts[cls_id] += 1
            except Exception:
                continue
        
        print(f"  {split_name} ({total_samples} samples):")
        for cls_id in sorted(split_class_counts.keys()):
            count = split_class_counts[cls_id]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"    Class {cls_id}: {count} samples ({percentage:.1f}%)")
