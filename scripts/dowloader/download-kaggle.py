import kagglehub
import shutil
import os


def get_data_from_kaggle(dataset_name, target_folder):
    if not os.path.exists(target_folder):
        # make directory if it doesn't exist
        os.makedirs(target_folder)
    # Download dataset
    path = kagglehub.dataset_download(dataset_name)
    # Copy to your local project folder
    shutil.copytree(path, target_folder, dirs_exist_ok=True)
    # Remove temp folder
    shutil.rmtree(path)


if __name__ == "__main__":
    # https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format
    get_data_from_kaggle(
        "sapal6/waste-classification-data-v2", "./data/waste-classification-v2"
    )
