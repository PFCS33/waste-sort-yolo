from datetime import datetime
import os
import wandb
from ultralytics import YOLO, settings


def train(model, config, pretrained_weight_path=None):
    # load pretrained weight
    if pretrained_weight_path is not None:
        model.load(pretrained_weight_path)
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
    )
    metrics = {
        "mAP50": results.results_dict["metrics/mAP50(B)"],
        "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
        "precision": results.results_dict["metrics/precision(B)"],
        "recall": results.results_dict["metrics/recall(B)"],
    }
    wandb.summary.update(metrics)

    # finish wandb
    wandb.finish()
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
    wandb.summary.update(
        {
            "test/mAP50": results.box.map50,
            "test/mAP50-95": results.box.map,
            "test/precision": results.box.mp,
            "test/recall": results.box.mr,
        }
    )
    wandb.finish()
