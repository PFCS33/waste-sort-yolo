from ultralytics import  settings
import os
import wandb


def set_settings(root_dir):
    settings.update(
        {
            "datasets_dir": os.path.join(root_dir, "data"),
            "runs_dir": os.path.join(root_dir, "runs"),
            "weights_dir": os.path.join(root_dir, "weights"),
            "wandb": True
        }
    )
    print(f"settings set:\n",settings) 


def wandb_login():
    wandb.login(key='bf880318d74672f2495c07b682f715d64df89d39')