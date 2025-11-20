from datetime import datetime
import wandb


def train(model, config):
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    # wandb init
    wandb.init(
        project="waste-sorting",
        name=run_name,
        tags=config["tags"],
        config={
            "model_type": config["model"],
            "pretrain_weight": config["pretrain_weight"],
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

    # finish wandb
    wandb.finish()
    return results
`