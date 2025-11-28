from datetime import datetime
import os
import wandb
from ultralytics import YOLO, settings


def train(model, config, weight_path=None):
    # load pretrained weight
    if weight_path is not None:
        model.load(weight_path)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    # wandb init
    wandb.init(
        project=config["project"],
        name=run_name,
        tags=config["tags"],
        config={
            "model_type": config["model"],
            "pretrain_weight": config["pretrained_weight"],
            "dataset": config["data_path"],
        },
    )
    # train
    results = model.train(
        name=run_name,
        data=config["data_path"],
        epochs=config["num_epochs"],
        batch=config["batch_size"],
        imgsz=config["image_size"],
        device=config["device"],
        workers=config["workers"],
        patience=config["patience"],
        # lr0=config["lr0"],
    )
    metrics = {
        "mAP50": results.results_dict["metrics/mAP50(B)"],
        "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
        "precision": results.results_dict["metrics/precision(B)"],
        "recall": results.results_dict["metrics/recall(B)"],
    }
    print("\n" + "="*50)
    print("Final Training Metrics:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50 + "\n")
    return results


def test(run_name, config):
    wandb.init(project=config["project"], name=f"${run_name}_test", tags="test")
    model_path = os.path.join(
        settings["runs_dir"], "detect", run_name, "weights", "best.pt"
    )
    model = YOLO(model_path)
    results = model.val(
        data=config["data_path"],
        batch=config["batch_size"],
        imgsz=config["image_size"],
        device=config["device"],
    )
    print("\n" + "="*50)
    print("Test Metrics:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50 + "\n")
