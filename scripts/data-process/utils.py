import os
import shutil
import kagglehub
from roboflow import Roboflow
import yaml
import json
import cv2

def load_config(config_path):
    """Read dataset configuration from YAML"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------- download --------------------------------- #
def download_all(config):
    """Download all datasets specified in config file"""
    root_dir = config["global"]["root_dir"]

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        download_config = dataset_config.get("download")
        if not download_config:
            print(f"No download config for dataset: {dataset_name}")
            continue

        # Create output directory: root_dir/dataset_name
        output_dir = os.path.join(root_dir, dataset_name)

        # Delete old dataset directory if it exists
        if os.path.exists(output_dir):
            print(f"Removing existing dataset directory: {output_dir}")
            shutil.rmtree(output_dir)

        source_type = download_config["source"]

        # Handle different parameter formats in config
        if "params" in download_config:
            params = download_config["params"]
        else:
            print(f"No params found for dataset: {dataset_name}")
            continue

        print(f"Downloading {dataset_name} from {source_type} to {output_dir}...")

        try:
            download(output_dir, source_type, params)
            print(f"✓ Successfully downloaded {dataset_name}")
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")


def download(output_dir, source_type, params):
    """
    Unified downloader

    Args:
        output_dir (str): Target directory for downloaded dataset
        source_type (str): 'kaggle' | 'roboflow'
        params (dict): Source-specific parameters

    Returns:
        str: Path to downloaded dataset
    """
    if source_type == "kaggle":
        return download_kaggle(output_dir, params)
    elif source_type == "roboflow":
        return download_roboflow(output_dir, params)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def download_kaggle(output_dir, params):
    """
    Download dataset from Kaggle

    Args:
        output_dir (str): Target directory
        params (list): [0] dataset anme

    Returns:
        str: Path to downloaded dataset
    """

    dataset_name = params[0]
    if not dataset_name:
        raise ValueError("dataset_name is required for Kaggle downloads")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download dataset
    path = kagglehub.dataset_download(dataset_name)

    # Copy to target folder
    shutil.copytree(path, output_dir, dirs_exist_ok=True)

    # Remove temp folder
    shutil.rmtree(path)

    return output_dir


def download_roboflow(output_dir, params):
    """
    Download dataset from Roboflow

    Args:
        output_dir (str): Target directory
        params (list): e.g. ["object-detection-cfmul", "waste-detection-0momv", 7, "yolov8"]

    Returns:
        str: Path to downloaded dataset
    """
    api_key = "u6dBxE6mzDkitzf9IOgk"
    workspace = params[0]
    project = params[1]
    version = params[2]
    dformat = params[3]

    if not all([api_key, workspace, project, version]):
        raise ValueError(
            "api_key, workspace, project, and version are required for Roboflow downloads"
        )

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)

    version_obj.download(dformat, location=output_dir)

    return output_dir


# --------------------------------- transform -------------------------------- #
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
        print(f"Dataset root: {dataset_root}")

        # Delete old transformed directory if it exists
        transformed_dir = os.path.join(dataset_root, "transformed")
        if os.path.exists(transformed_dir):
            print(f"Removing existing transformed directory: {transformed_dir}")
            shutil.rmtree(transformed_dir)

        # Extract transform parameters
        cls_mapping = transform_config.get("class_mapping", {})
        coord_transform = transform_config.get("coordinate_transform")
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

        try:
            # Call the existing transform function
            transform(dataset_root, source_pairs, cls_mapping, coord_transform)
            print(f"✓ Successfully transformed {dataset_name}")
        except Exception as e:
            print(f"✗ Failed to transform {dataset_name}: {e}")


def transform(dataset_root, source_pairs, cls_mapping, coord_transform=None):
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
        print(f"  Images: {images_dir}")
        print(f"  Labels: {labels_dir}")

        print(
            f"Transforming images from {images_dir} (starting at index {current_index})..."
        )
        rename_mapping, next_index = image_transform(
            images_dir, images_target, current_index
        )

        print(f"Transforming labels from {labels_dir}...")
        label_transform(
            labels_dir, labels_target, cls_mapping, coord_transform, rename_mapping
        )

        processed_count = next_index - current_index
        total_processed += processed_count
        current_index = next_index

        print(f"Processed {processed_count} files from this source pair")

    print(f"\nTransformation complete! Processed {total_processed} total files.")
    print(f"Results saved in {transformed_dir}")
    print(f"- Images: {images_target}")
    print(f"- Labels: {labels_target}")


def image_transform(source, target, start_index=0):
    if not os.path.exists(source):
        print(f"Error: Source directory {source} does not exist")
        return [], start_index

    os.makedirs(target, exist_ok=True)

    # get all image files and sort them for consistent order
    image_files = [
        f for f in os.listdir(source) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    if len(image_files) == 0:
        print(f"No image files found in {source}")
        return [], start_index

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

    print(
        f"Copied and renamed {len(image_files)} images from {source} to {target} (indices {start_index}-{next_index-1})"
    )
    print(f"Saved rename mapping to {mapping_path}")
    return rename_mapping, next_index


def label_transform(
    source, target, cls_mapping, coord_transform=None, rename_mapping=None
):
    if not os.path.exists(source):
        print(f"Error: Source directory {source} does not exist")
        return

    os.makedirs(target, exist_ok=True)

    # 1. read all txt files under source/, with consistent processing order of images
    txt_files = [f for f in os.listdir(source) if f.endswith(".txt")]
    txt_files.sort()

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
                continue

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

    print(f"Transformed {len(txt_files)} label files from {source} to {target}")


def coord_transform(x_center, y_center, width, height):
    return x_center, y_center, width, height


# ----------------------------------- other ---------------------------------- #
def draw_label(image_path, label_path):
    # 1. read image and label(txt)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    h, w = image.shape[:2]

    try:
        with open(label_path, "r") as f:
            first_line = f.readline().strip()
    except FileNotFoundError:
        print(f"Error: Could not load label file from {label_path}")
        return

    # 2. draw bbox on image (follow the format that cls_id x_center ycenter width height)
    if first_line:
        parts = first_line.split()
        if len(parts) != 5:
            print(f"Error: Invalid label format in first line: {first_line}")
            return

        cls_id, x_center, y_center, width, height = map(float, parts)
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # draw bbox
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # add class label
        label_text = f"Class {int(cls_id)}"
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )

    # 3. show img with bbox
    cv2.imshow("Image with Annotations", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def count_images(image_dir):

    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist")
        return 0

    # Count image files
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
    ]
    count = len(image_files)

    print(f"Total images in {image_dir}: {count}")
    return count



