import os
import argparse
from ultralytics import YOLO
from scripts.utils import *
from scripts.train import *

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


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="mode", help="Available modes: train / test / convert", required=True
    )

    # Train subparser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--run_name", type=str, help="Run name to load last.pt weights from"
    )
    # python3 main.py train --run_name 

    # Test subparser
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("run_name", type=str)

    # Convert subparser
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--path", type=str, required=True, help="path to .pt file you want to convert")

    return parser.parse_args()


def main():
    args = parse_args()

    # initial settings
    wandb_login()
    set_settings(ROOT_DIR)

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
    
    elif args.mode == "convert":
        # convert model to TensorFlow Lite
        tflite_path = convert_to_tf(args.path)
        if tflite_path:
            print(f"Conversion successful! TFLite model saved at: {tflite_path}")
        else:
            print("Conversion failed!")


if __name__ == "__main__":
    main()
