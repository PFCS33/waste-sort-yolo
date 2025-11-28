import os
import shutil
import json


def transform_all(config):
    """Transform all datasets specified in config file"""
    root_dir = config["global"]["root_dir"]

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        transform_config = dataset_config.get("transform")
        if not transform_config:
            print(f"No transform config for dataset: {dataset_name}")
            continue

        # Build dataset directory path
        dataset_root = os.path.join(root_dir, dataset_name)
        if not os.path.exists(dataset_root):
            print(f"Dataset directory {dataset_root} does not exist, skipping...")
            continue

        print(f"\nTransforming dataset: {dataset_name}")

        # Delete old transformed directory if it exists
        transformed_dir = os.path.join(dataset_root, "transformed")
        if os.path.exists(transformed_dir):
            print(f"Removing existing transformed directory: {transformed_dir}")
            shutil.rmtree(transformed_dir)

        # Extract transform parameters
        cls_mapping = transform_config.get("class_mapping", {})
        coord_transform = transform_config.get("coordinate_transform")
        exclude_class = transform_config.get("exclude_class")
        paths_config = transform_config.get("paths", [])

        if not paths_config:
            print(f"No paths specified for dataset: {dataset_name}")
            continue

        # Convert paths config to source_pairs format expected by transform()
        source_pairs = []
        for path_config in paths_config:
            if "images" in path_config and "labels" in path_config:
                # Build full paths by joining dataset_root with path segments
                images_path_segments = path_config["images"]
                labels_path_segments = path_config["labels"]

                images_path = os.path.join(dataset_root, *images_path_segments)
                labels_path = os.path.join(dataset_root, *labels_path_segments)

                source_pairs.append({"images": images_path, "labels": labels_path})

        if not source_pairs:
            print(f"No valid source pairs found for dataset: {dataset_name}")
            continue

        exclude_class_set = set(exclude_class) if exclude_class else None
        try:
            transform(
                dataset_root,
                source_pairs,
                cls_mapping,
                coord_transform,
                exclude_class_set,
            )
            print(f"✓ Successfully transformed {dataset_name}")
        except Exception as e:
            print(f"✗ Failed to transform {dataset_name}: {e}")


def transform(
    dataset_root,
    source_pairs,
    cls_mapping,
    coord_transform=None,
    exclude_classes=None,
):
    # read all files from image/label directory pairs
    # 1. for images, call image_transform, move it to transformed/images with continuous numbering
    # 2. for txt(label), call label_transform: transform each label, and move it under transformed/labels

    if not source_pairs:
        print("Error: No source directory pairs provided")
        return

    # different input formats for items inside the list
    normalized_pairs = []
    for item in source_pairs:
        if isinstance(item, str):
            # format1: string directory (assuming both images and labels are there)
            normalized_pairs.append({"images": item, "labels": item})
        elif isinstance(item, dict):
            # format2: {"images": "path", "labels": "path"}
            normalized_pairs.append(item)
        else:
            print(f"Error: Invalid source pair format: {item}")
            continue

    # transformed directory under dataset_root
    transformed_dir = os.path.join(dataset_root, "transformed")
    images_target = os.path.join(transformed_dir, "images")
    labels_target = os.path.join(transformed_dir, "labels")
    os.makedirs(images_target, exist_ok=True)
    os.makedirs(labels_target, exist_ok=True)

    current_index = 0
    total_processed = 0
    total_excluded = 0

    for i, source_pair in enumerate(normalized_pairs):
        images_dir = source_pair["images"]
        labels_dir = source_pair["labels"]

        if not os.path.exists(images_dir):
            print(f"Error: Images directory {images_dir} does not exist, skipping...")
            continue

        if not os.path.exists(labels_dir):
            print(f"Error: Labels directory {labels_dir} does not exist, skipping...")
            continue

        print(f"\nProcessing source pair {i+1}/{len(normalized_pairs)}")

        # build excluded image names set for this source pair
        excluded_image_names = build_excluded_image_names(labels_dir, exclude_classes)

        rename_mapping, next_index, excluded_count = image_transform(
            images_dir, images_target, current_index, excluded_image_names
        )

        label_transform(
            labels_dir,
            labels_target,
            cls_mapping,
            coord_transform,
            rename_mapping,
            excluded_image_names,
        )

        processed_count = next_index - current_index
        total_processed += processed_count
        total_excluded += excluded_count
        current_index = next_index

    print(f"\nTransformation complete! Processed {total_processed} total files.")
    if total_excluded > 0:
        print(
            f"Excluded {total_excluded} files containing forbidden classes: {list(exclude_classes)}"
        )
    print(f"Results saved in {transformed_dir}")


def image_transform(source, target, start_index=0, excluded_image_names=None):
    if not os.path.exists(source):
        print(f"Error: Source directory {source} does not exist")
        return [], start_index, 0

    os.makedirs(target, exist_ok=True)

    # get all image files and sort them for consistent order
    all_image_files = [
        f for f in os.listdir(source) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    all_image_files.sort()

    # Filter out excluded images if exclusion set provided
    if excluded_image_names:
        image_files = []
        excluded_count = 0
        for img_file in all_image_files:
            img_base_name = os.path.splitext(img_file)[0]
            if img_base_name in excluded_image_names:
                excluded_count += 1
            else:
                image_files.append(img_file)
    else:
        image_files = all_image_files
        excluded_count = 0

    if len(image_files) == 0:
        if excluded_count > 0:
            print(
                f"No valid image files found in {source} (all {excluded_count} images excluded)"
            )
        else:
            print(f"No image files found in {source}")
        return [], start_index, excluded_count

    rename_mapping = []
    for i, old_filename in enumerate(image_files):
        old_path = os.path.join(source, old_filename)
        file_ext = os.path.splitext(old_filename)[1]
        new_index = start_index + i
        new_filename = f"{new_index}{file_ext}"
        new_path = os.path.join(target, new_filename)

        # copy and rename the file
        shutil.copy2(old_path, new_path)
        rename_mapping.append((old_filename, new_filename))

    next_index = start_index + len(image_files)

    # save rename mapping to JSON file for this batch (in parent folder of target)
    target_parent = os.path.dirname(target)
    mapping_path = os.path.join(target_parent, f"{start_index}_rename_mapping.json")
    mapping_dict = {old: new for old, new in rename_mapping}
    batch_info = {
        "source": source,
        "start_index": start_index,
        "end_index": next_index - 1,
        "mapping": mapping_dict,
    }
    with open(mapping_path, "w") as f:
        json.dump(batch_info, f, indent=2)

    print(f"Saved rename mapping to {mapping_path}")
    return rename_mapping, next_index, excluded_count


def label_transform(
    source,
    target,
    cls_mapping,
    coord_transform=None,
    rename_mapping=None,
    excluded_image_names=None,
):
    if not os.path.exists(source):
        print(f"Error: Source directory {source} does not exist")
        return

    os.makedirs(target, exist_ok=True)

    # 1. read all txt files under source/, with consistent processing order of images
    all_txt_files = [f for f in os.listdir(source) if f.endswith(".txt")]
    all_txt_files.sort()

    # Filter out excluded labels if exclusion set provided
    if excluded_image_names:
        txt_files = []
        excluded_labels_count = 0
        for txt_file in all_txt_files:
            txt_base_name = os.path.splitext(txt_file)[0]
            if txt_base_name in excluded_image_names:
                excluded_labels_count += 1
                # Skip this label file as its corresponding image was excluded
            else:
                txt_files.append(txt_file)
    else:
        txt_files = all_txt_files
        excluded_labels_count = 0

    # create mapping from image basenames to new names for label renaming
    label_rename_map = {}
    if rename_mapping:
        for old_img, new_img in rename_mapping:
            old_base = os.path.splitext(old_img)[0]  # Remove extension
            new_base = os.path.splitext(new_img)[0]  # Remove extension
            label_rename_map[f"{old_base}.txt"] = f"{new_base}.txt"

    for txt_file in txt_files:
        source_path = os.path.join(source, txt_file)

        # Use renamed filename if mapping exists, otherwise use original
        target_filename = label_rename_map.get(txt_file, txt_file)
        target_path = os.path.join(target, target_filename)

        try:
            with open(source_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Could not read {source_path}")
            continue

        transformed_lines = []

        # 2. each file, read each line, process (cls_id x_center y_center width height)
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                # convert polygon
                cls_id = float(parts[0])
                x_center, y_center, width, height = polygon_to_yolo_bbox([float(p) for p in parts[1:]])
            else:
                cls_id, x_center, y_center, width, height = map(float, parts)

            # 3. class mapping
            if cls_mapping and int(cls_id) in cls_mapping:
                cls_id = cls_mapping[int(cls_id)]

            # 4. if have coord transform, apply, get new coordinates
            if coord_transform:
                x_center, y_center, width, height = coord_transform(
                    x_center, y_center, width, height
                )

            # 5. write back transformed line
            transformed_line = f"{int(cls_id)} {x_center} {y_center} {width} {height}\n"
            transformed_lines.append(transformed_line)

        # Write transformed labels to target
        with open(target_path, "w") as f:
            f.writelines(transformed_lines)


def coord_transform(x_center, y_center, width, height):
     return x_center, y_center, width, height



def polygon_to_yolo_bbox(polygon):
    """Convert polygon points to YOLO bbox format."""
    xs = polygon[0::2]  # Even indices: x coordinates
    ys = polygon[1::2]  # Odd indices: y coordinates

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def build_excluded_image_names(labels_dir, exclude_classes=None):
    """Scan labels directory and build set of image names that contain excluded classes"""
    if not exclude_classes or not os.path.exists(labels_dir):
        return set()
    excluded_image_names = set()

    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            # Check if any line contains excluded class
            has_excluded = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    cls_id = int(float(parts[0]))
                    if cls_id in exclude_classes:
                        has_excluded = True
                        break

            if has_excluded:
                # Get corresponding image name (without .txt extension)
                image_base_name = os.path.splitext(label_file)[0]
                excluded_image_names.add(image_base_name)

        except Exception:
            continue  # Skip problematic label files

    return excluded_image_names
