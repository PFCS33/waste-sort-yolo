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
    print("\n" + "=" * 50)
    print("Final Training Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50 + "\n")
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
    metrics = {
        "mAP50": results.results_dict["metrics/mAP50(B)"],
        "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
        "precision": results.results_dict["metrics/precision(B)"],
        "recall": results.results_dict["metrics/recall(B)"],
    }
    print("\n" + "=" * 50)
    print("Test Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50 + "\n")


def predict(weight_path, source, conf, save=True, show=False):
    model = YOLO(weight_path)

    print(f"Model classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    print(f"Confidence threshold: {conf}")
    print(f"Input source: {source}")

    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        verbose=True,
        visualize=True,
    )

    # Print detection results
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Original image shape: {result.orig_shape}")

        if result.boxes is not None:
            num_boxes = len(result.boxes)
            print(f"Number of detected boxes: {num_boxes}")

            if num_boxes > 0:
                print("Confidences:", result.boxes.conf.cpu().numpy())
                print("Classes:", result.boxes.cls.cpu().numpy())
                print(
                    "Class names:", [model.names[int(cls)] for cls in result.boxes.cls]
                )
                print("Boxes (xyxy):", result.boxes.xyxy.cpu().numpy())
            else:
                print("No boxes detected!")
        else:
            print("No boxes detected!")

    if show:
        for result in results:
            result.show()

    return results
