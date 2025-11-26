import os
import shutil
import random
from collections import defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
from .test import print_distributions


def merge_all(config, split=[0.8, 0.1, 0.1]):
    """
    Merge all transformed datasets and split with multi-label stratification.
    """
    root_dir = config["global"]["root_dir"]
    custom_name = config["global"]["custom_name"]

    # use num_classes from config if available
    num_classes = config["global"]["nc"]

    print(f"Starting merge_all: merging datasets into {custom_name}")

    # 1: Collect transformed datasets
    transformed_datasets = collect_transformed_datasets(config)
    if not transformed_datasets:
        print("No transformed datasets found!")
        return

    # 2: Setup merge directory
    merge_dir = os.path.join(root_dir, custom_name)
    origin_dir = os.path.join(merge_dir, "origin")
    origin_images = os.path.join(origin_dir, "images")
    origin_labels = os.path.join(origin_dir, "labels")

    if os.path.exists(merge_dir):
        shutil.rmtree(merge_dir)
    os.makedirs(origin_images, exist_ok=True)
    os.makedirs(origin_labels, exist_ok=True)

    # 3: Merge all datasets
    merge_datasets(transformed_datasets, origin_images, origin_labels)

    # 4: Analyze with multi-label awareness
    sample_ids, class_matrix, class_counts = calc_class_distribution(
        origin_labels, num_classes=num_classes
    )

    # 5: Create balanced splits using multi-label stratification
    stratified_multilabel_splits(merge_dir, origin_dir, sample_ids, class_matrix, split)

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

        skipped_count = 0
        for image_file in image_files:
            # Check if corresponding label exists first
            label_file = os.path.splitext(image_file)[0] + ".txt"
            old_label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(old_label_path):
                print(f"Warning: Label not found for {image_file}, skipping...")
                skipped_count += 1
                continue

            # Copy image with new index
            old_image_path = os.path.join(images_dir, image_file)
            file_ext = os.path.splitext(image_file)[1]
            new_image_name = f"{current_index}{file_ext}"
            new_image_path = os.path.join(target_images, new_image_name)
            shutil.copy2(old_image_path, new_image_path)

            # Copy corresponding label
            new_label_name = f"{current_index}.txt"
            new_label_path = os.path.join(target_labels, new_label_name)
            shutil.copy2(old_label_path, new_label_path)

            current_index += 1

        merged_count = len(image_files) - skipped_count
        total_merged += merged_count
        if skipped_count > 0:
            print(
                f"  ✓ Merged {merged_count} samples from {dataset['name']} (skipped {skipped_count} without labels)"
            )
        else:
            print(f"  ✓ Merged {merged_count} samples from {dataset['name']}")

    print(f"Total samples merged: {total_merged}")
    return total_merged


def calc_class_distribution(labels_dir, num_classes=None):
    """
    Analyze class distribution and build multi-hot matrix for stratified splitting.

    Returns:
        sample_ids: [sample id]
        class_matrix: np array, multi-hot encoding of (n_samples, n_classes)
        class_counts: dict of {class_id: count}
    """
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])

    # find all classes and collect sample data
    sample_cls_map = {}  # sample_id -> set of classes
    all_classes = set()

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        sample_id = os.path.splitext(label_file)[0]

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            classes_per_sample = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    cls_id = int(float(parts[0]))
                    classes_per_sample.add(cls_id)
                    all_classes.add(cls_id)

            if classes_per_sample:
                sample_cls_map[sample_id] = classes_per_sample

        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue

    # determine total number of classes
    if num_classes is None:
        num_classes = max(all_classes) + 1 if all_classes else 0

    # Build multi-hot matrix for multi-class per sample case
    sample_ids = list(sample_cls_map.keys())
    class_matrix = np.zeros((len(sample_ids), num_classes), dtype=np.int8)

    for i, sample_id in enumerate(sample_ids):
        for cls_id in sample_cls_map[sample_id]:
            if cls_id < num_classes:
                class_matrix[i, cls_id] = 1

    # calculate class counts
    class_counts = defaultdict(int)
    for cls_id in range(num_classes):
        class_counts[cls_id] = int(class_matrix[:, cls_id].sum())

    # Print distribution
    print("\nClass distribution:")
    sample_id_range = len(sample_ids)
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        if count > 0:
            percentage = (count / sample_id_range) * 100
            print(f"  Class {cls_id}: {count} samples ({percentage:.1f}%)")

    return sample_ids, class_matrix, dict(class_counts)


def stratified_multilabel_splits(merge_dir, origin_dir, sample_ids, class_matrix, split):
    """
    create balanced train/val/test splits using multi-label stratification.

    Args:
        merge_dir: Output directory for splits
        origin_dir: Source directory with merged images/labels
        sample_ids: [sample id]
        class_matrix: Multi-hot encoding matrix (n_samples, n_classes)
        split: [train_ratio, val_ratio, test_ratio]
    """
    train_ratio, val_ratio, test_ratio = split

    # clean up old split directories
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(merge_dir, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)

    origin_images = os.path.join(origin_dir, "images")
    origin_labels = os.path.join(origin_dir, "labels")

    sample_ids = np.array(sample_ids)
    sample_id_range = np.arange(len(sample_ids))

    # train vs (val + test) split
    temp_ratio = val_ratio + test_ratio

    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=temp_ratio, random_state=42
    )
    train_idx, temp_idx = next(msss1.split(sample_id_range, class_matrix))

    # val vs test split
    relative_test_ratio = test_ratio / temp_ratio

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_ratio, random_state=42
    )
    val_idx_relative, test_idx_relative = next(
        msss2.split(temp_idx, class_matrix[temp_idx])
    )

    # convert relative indices back to absolute
    val_idx = temp_idx[val_idx_relative]
    test_idx = temp_idx[test_idx_relative]

    split_assignments = {
        "train": sample_ids[train_idx].tolist(),
        "val": sample_ids[val_idx].tolist(),
        "test": sample_ids[test_idx].tolist(),
    }

    # Move files to appropriate split directories
    for split_name, sample_list in split_assignments.items():
        print(f"\nMoving {len(sample_list)} samples to {split_name}...")

        target_images = os.path.join(merge_dir, split_name, "images")
        target_labels = os.path.join(merge_dir, split_name, "labels")

        for sample_id in sample_list:
            # Move image
            for ext in [".jpg", ".jpeg", ".png"]:
                origin_img_path = os.path.join(origin_images, f"{sample_id}{ext}")
                if os.path.exists(origin_img_path):
                    shutil.move(
                        origin_img_path,
                        os.path.join(target_images, f"{sample_id}{ext}"),
                    )
                    break

            # Move label
            origin_label_path = os.path.join(origin_labels, f"{sample_id}.txt")
            if os.path.exists(origin_label_path):
                shutil.move(
                    origin_label_path, os.path.join(target_labels, f"{sample_id}.txt")
                )

    # Clean up origin directory
    if os.path.exists(origin_dir):
        shutil.rmtree(origin_dir)

    # Print final distribution per split
    print_distributions(merge_dir)


