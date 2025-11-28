import os
import cv2
from collections import defaultdict


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


def print_distributions(merge_dir, config):
    """print class distribution for each split and return the distribution data"""
    print("\n" + "=" * 50)
    print("Final class distribution in each split:")
    print("=" * 50)
    
    # Read class names from config
    class_names = {}
    if config and "global" in config and "classes" in config["global"]:
        classes_list = config["global"]["classes"]
        for i, class_name in enumerate(classes_list):
            class_names[i] = class_name

    # Prepare content for return
    output_lines = []
    output_lines.append("=" * 50)
    output_lines.append("Final class distribution in each split:")
    output_lines.append("=" * 50)

    for split_name in ["train", "val", "test"]:
        split_labels = os.path.join(merge_dir, split_name, "labels")
        label_files = [f for f in os.listdir(split_labels) if f.endswith(".txt")]
        total_samples = len(label_files)

        # Count all classes (not just primary)
        class_counts = defaultdict(int)
        for label_file in label_files:
            label_path = os.path.join(split_labels, label_file)
            try:
                with open(label_path, "r") as f:
                    classes_seen = set()
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls_id = int(float(parts[0]))
                            classes_seen.add(cls_id)
                    for cls_id in classes_seen:
                        class_counts[cls_id] += 1
            except Exception:
                continue

        split_header = f"\n{split_name.upper()} ({total_samples} samples):"
        print(split_header)
        output_lines.append(split_header)
        
        for cls_id in sorted(class_counts.keys()):
            count = class_counts[cls_id]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            
            # Get class name or use fallback
            class_name = class_names.get(cls_id, "?")
            line = f"  Class {cls_id} ({class_name}): {count} ({percentage:.1f}%)"
            print(line)
            output_lines.append(line)
    
    return "\n".join(output_lines)
