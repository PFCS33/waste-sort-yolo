from utils import test_annoation, count_images, transform_source
import os
import argparse

DATASET_ROOT = "data/Waste-Detection-7"


def main():
    # Create argument parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # check whether bbox follow the (x_center, y_center, width, height) format
    label_check_parser = subparsers.add_parser(
        "label-check", help="check and visualize labels"
    )

    transform_parset = subparsers.add_parser(
        "transform", help="trasnform images & label from source to target"
    )

    count_parser = subparsers.add_parser("count", help="print out number of images")

    args = parser.parse_args()

    if args.mode == "label-check":
        image_path = os.path.join(
            DATASET_ROOT,
            "transformed",
            "images",
            "3475.jpg",
        )
        label_path = os.path.join(
            DATASET_ROOT,
            "transformed",
            "labels",
            "3475.txt",
        )
        test_annoation(image_path, label_path)
    elif args.mode == "transform":
        source_dirs = [
            {
                "images": os.path.join(DATASET_ROOT, "train", "images"),
                "labels": os.path.join(DATASET_ROOT, "train", "labels"),
            },
            {
                "images": os.path.join(DATASET_ROOT, "valid", "images"),
                "labels": os.path.join(DATASET_ROOT, "valid", "labels"),
            },
            {
                "images": os.path.join(DATASET_ROOT, "test", "images"),
                "labels": os.path.join(DATASET_ROOT, "test", "labels"),
            },
        ]
        transform_source(
            DATASET_ROOT, source_dirs, {0: 5, 1: 1, 2: 10, 3: 2, 4: 5, 5: 0}
        )

    elif args.mode == "count":
        image_dir = os.path.join(DATASET_ROOT, "train", "images")
        count_images(image_dir)


if __name__ == "__main__":
    main()
