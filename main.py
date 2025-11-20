import os
from ultralytics import YOLO, settings
from scripts.utils import *
from scripts.train import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CONFIG = {
    "model": "yolov8n.pt",
    "pretrain_weight":"yolov8n.pt",
    "tags": ["yolov8n", "baseline"],
    "data_path": os.path.join(
        ROOT_DIR, "data", "GARBAGE-CLASSIFICATION-3-2", "data.yaml"
    ),
    "num_epochs": 50,
    "batch_size": 16,
    "image_size": 640,
    "device": 0,  # GPU ID
    "workers": 8,
    "patience": 20
}

wandb_login()
set_settings(ROOT_DIR)


model = YOLO(TRAIN_CONFIG["model"])
model.load(TRAIN_CONFIG['pretrain_weight'])  # load pretrained COCO weights

train(
    model,
    TRAIN_CONFIG,
)
