import argparse
from utils import (
    download_all,
    load_config,
    transform_all,
    merge_all,
    draw_label,
    count_images,
    print_distributions,
)

# path to config yaml file
DATASET_CONFIG_PATH = "dataset-config.yaml"


def process_all(config):
    """Process all steps: download and transform"""
    pass
    # download_all(config_path)
    # transform_all(config_path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    # Download subparser
    download_parser = subparsers.add_parser(
        "download", help="Only download all datasets"
    )
    download_parser.add_argument(
        "--config",
        default=DATASET_CONFIG_PATH,
        help="Path to configuration yaml file",
    )
    # Transform subparser
    transform_parser = subparsers.add_parser("transform", help="Transform datasets")
    transform_parser.add_argument(
        "--config",
        default=DATASET_CONFIG_PATH,
        help="Path to configuration yaml file",
    )
    # Merge subparser
    merge_parser = subparsers.add_parser("merge", help="Merge transformed datasets")
    merge_parser.add_argument(
        "--config",
        default=DATASET_CONFIG_PATH,
        help="Path to configuration yaml file",
    )
    # All-in-one subparser
    all_parser = subparsers.add_parser("all", help="Download and transform datasets")
    all_parser.add_argument(
        "--config",
        default=DATASET_CONFIG_PATH,
        help="Path to configuration yaml file",
    )
    # Test subparser
    test_parser = subparsers.add_parser(
        "test", help="Test datasets (draw_label or count_images)"
    )
    test_parser.add_argument(
        "--config",
        default=DATASET_CONFIG_PATH,
        help="Path to configuration yaml file",
    )
    test_parser.add_argument(
        "--func",
        choices=["draw", "count", "distribution"],
        required=True,
        help="Test mode: draw | count | distribution",
    )
    test_parser.add_argument(
        "--image-path",
        help="Path to image file (required for draw)",
    )
    test_parser.add_argument(
        "--label-path",
        help="Path to label file (required for draw)",
    )
    test_parser.add_argument(
        "--image-dir",
        help="Path to images directory (required for count)",
    )
    test_parser.add_argument(
        "--merge-dir",
        help="Path to merged dataset directory (required for distribution)",
    )

    args = parser.parse_args()
    config_path = getattr(args, "config", DATASET_CONFIG_PATH)
    config = load_config(config_path)

    if args.mode == "download":
        download_all(config)
    elif args.mode == "transform":
        transform_all(config)
    elif args.mode == "merge":
        merge_all(config)
    elif args.mode == "all":
        process_all(config)
    elif args.mode == "test":
        if args.func == "draw":
            if not args.image_path or not args.label_path:
                print("Error: draw_label requires --image-path and --label-path")
                return
            draw_label(args.image_path, args.label_path)
        elif args.func == "count":
            if not args.image_dir:
                print("Error: count_images requires --image-dir")
                return
            count_images(args.image_dir)
        elif args.func == "distribution":
            if not args.merge_dir:
                print("Error: distribution requires --merge_dir")
                return
            print_distributions(args.merge_dir)


if __name__ == "__main__":
    main()
