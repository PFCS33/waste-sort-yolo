"""Common Utils Functions"""

from ultralytics import settings
import os
import wandb
from ultralytics import YOLO


def set_yolo_settings(root_dir):
    settings.update(
        {
            "datasets_dir": os.path.join(root_dir, "data"),
            "runs_dir": os.path.join(root_dir, "runs"),
            "weights_dir": os.path.join(root_dir, "weights"),
            "wandb": True,
        }
    )
    print(f"settings set:\n", settings)


def wandb_login():
    wandb.login(key="bf880318d74672f2495c07b682f715d64df89d39")


def convert_to_tf(model_path):
    # load weights from path
    # convert to tf-lite for mobile deployment
    # save it under same folder
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        return None

    # Load the model
    model = YOLO(model_path)

    # Export to TensorFlow Lite
    try:
        tflite_path = model.export(format="tflite", int8=True)
        print(f"Model successfully converted to TensorFlow Lite")
        print(f"Original model: {model_path}")
        print(f"TFLite model saved at: {tflite_path}")
        return tflite_path
    except Exception as e:
        print(f"Error converting model to TensorFlow Lite: {e}")
        return None


