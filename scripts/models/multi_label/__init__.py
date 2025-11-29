"""
Multi-label Detection for YOLO.
"""

from datetime import datetime
import wandb

from .config import HierarchyConfig
from .dataset import HierarchicalYOLODataset
from .loss import HierarchicalDetectionLoss
from .nms import hierarchical_nms, _dixon_q_test
from .predictor import HierarchicalDetectionPredictor
from .trainer import HierarchicalDetectionTrainer


def train(config, h_config_path):

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # WandB init
    wandb.init(
        project=config["project"],
        name=run_name,
        tags=config.get("tags", []),
        config={
            "model_type": config["model"],
            "pretrain_weight": config.get("pretrained_weight"),
            "dataset": config["data_path"],
            "hierarchical": True,
        },
    )

    overrides = {
        "model": config.get("pretrained_weight") or config["model"],
        "data": config["data_path"],
        "epochs": config["num_epochs"],
        "batch": config["batch_size"],
        "imgsz": config["image_size"],
        "device": config["device"],
        "workers": config["workers"],
        "patience": config["patience"],
        "name": run_name,
        "project": f"runs/{config['project']}",
        "hierarchy_config": h_config_path,
        "loss_type": config.get("loss_type", "bce"),
    }

    trainer = HierarchicalDetectionTrainer(overrides=overrides)
    results = trainer.train()

    metrics = {
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
    }

    print("\n" + "=" * 50)
    print("Final Training Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50 + "\n")

    wandb.finish()

    return results


def predict(model_path, image_path, h_config_path, conf=0.25, save=True, show=False):
    predictor = HierarchicalDetectionPredictor(
        overrides={
            "model": model_path,
            "hierarchy_config": h_config_path,
            "conf": conf,
        }
    )
    results = predictor(image_path)
    if save:
        for i, result in enumerate(results):
            save_path = f"prediction_{i}.jpg"
            result.save(save_path)
            print(f"Saved: {save_path}")

    if show:
        for result in results:
            result.show()
