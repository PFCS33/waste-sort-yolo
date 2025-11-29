import os
import argparse
from ultralytics import YOLO
from scripts.utils import *


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CONFIG = {
    "project": "waste-sorting",
    "model": "yolov8n.yaml",
    "pretrained_weight": os.path.join(
        ROOT_DIR, "weights", "yolov8n.pt"
    ),  # load pretrained COCO weights
    "tags": ["yolov8n", "baseline"],
    "data_path": os.path.join(
        ROOT_DIR, "data", "GARBAGE-CLASSIFICATION-3-2", "data.yaml"
    ),
    "num_epochs": 150,
    "batch_size": 16,
    "image_size": 640,
    "device": 0,  # GPU ID
    "workers": 8,
    "patience": 20,
    # "lr0": 0.001
}

#  Multi-label config
MULTI_LABEL_CONFIG = {
    "project": "waste-sorting-multi_lable",
    "model": "yolov8n.yaml",
    "pretrained_weight": os.path.join(ROOT_DIR, "weights", "yolov8n.pt"),
    "tags": ["yolov8n", "merge-data", "multi-label"],
    "data_path": os.path.join(
        ROOT_DIR,
        "data",
        "GARBAGE-CLASSIFICATION-3-2",
        "data_hierarchical.yaml",  # nc=19
    ),
    "config_file": os.path.join(ROOT_DIR, "scripts", "data", "config_h.yaml"),
    "num_epochs": 150,
    "batch_size": 16,
    "image_size": 640,
    "device": 0,
    "workers": 8,
    "patience": 20,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Waste Sorting Project - Unified Entry Point"
    )

    # Method selection
    parser.add_argument(
        "--method",
        choices=["baseline", "multi-label"],
        default="baseline",
        help="Choose implementation method (default: baseline)",
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        help="Available modes: train / test / predict / convert",
        required=True,
    )

    # Train subparser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--run_name", type=str, help="Run name to load last.pt weights from"
    )

    # Test subparser
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("run_name", type=str)

    # Convert subparser
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument(
        "--path", type=str, required=True, help="path to .pt file you want to convert"
    )

    # predict parser
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument(
        "--source", type=str, required=True, help="Image path, directory, or video"
    )
    predict_parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name to load best.pt weights from",
    )
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument(
        "--save",
        action="store_true",
    )
    predict_parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction results"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # initial settings
    wandb_login()
    set_yolo_settings(ROOT_DIR)

    if args.method == "baseline":
        from scripts.models.baseline import train, test, predict

        if args.mode == "train":
            # modify pretrained_weight if run_name provided
            if args.run_name:
                TRAIN_CONFIG["pretrained_weight"] = os.path.join(
                    ROOT_DIR, "runs", "detect", args.run_name, "weights", "last.pt"
                )

            # model
            model = YOLO(TRAIN_CONFIG["model"])
            # train
            train(model, TRAIN_CONFIG, weight_path=TRAIN_CONFIG["pretrained_weight"])

        elif args.mode == "test":
            # test
            test(args.run_name, TRAIN_CONFIG)

        elif args.mode == "predict":
            weight_path = os.path.join(
                ROOT_DIR, "runs", "detect", args.run_name, "weights", "best.pt"
            )
            predict(weight_path, args.source, args.conf, args.save, args.show)
        elif args.mode == "convert":
            # convert model to TensorFlow Lite
            tflite_path = convert_to_tf(args.path)
            if tflite_path:
                print(f"Conversion successful! TFLite model saved at: {tflite_path}")
            else:
                print("Conversion failed!")

    elif args.method == "multi-label":
        from scripts.models.multi_label import train, predict

        if args.mode == "train":
            if args.run_name:
                MULTI_LABEL_CONFIG["pretrained_weight"] = os.path.join(
                    ROOT_DIR, "runs", "detect", args.run_name, "weights", "last.pt"
                )

            train(
                config=MULTI_LABEL_CONFIG,
                h_config_path=MULTI_LABEL_CONFIG["config_file"],
            )

        elif args.mode == "predict":
            weight_path = os.path.join(
                ROOT_DIR, "runs", "detect", args.run_name, "weights", "best.pt"
            )

            predict(
                model_path=weight_path,
                image_path=args.source,
                h_config_path=MULTI_LABEL_CONFIG["config_file"],
                conf=args.conf,
                save=args.save,
                show=args.show,
            )
            pass
        elif args.mode == "convert":
            pass


if __name__ == "__main__":
    main()
