import os
import shutil
import kagglehub
from roboflow import Roboflow


def download_all(config):
    """Download all datasets specified in config file"""
    root_dir = config["global"]["root_dir"]

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        download_config = dataset_config.get("download")
        if not download_config:
            print(f"No download config for dataset: {dataset_name}")
            continue

        # Create output directory: root_dir/dataset_name
        output_dir = os.path.join(root_dir, dataset_name)

        # Delete old dataset directory if it exists
        if os.path.exists(output_dir):
            print(f"Removing existing dataset directory: {output_dir}")
            shutil.rmtree(output_dir)

        source_type = download_config["source"]

        # Handle different parameter formats in config
        if "params" in download_config:
            params = download_config["params"]
        else:
            print(f"No params found for dataset: {dataset_name}")
            continue

        print(f"Downloading {dataset_name} from {source_type} to {output_dir}...")

        try:
            download(output_dir, source_type, params)
            print(f"✓ Successfully downloaded {dataset_name}")
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")


def download(output_dir, source_type, params):
    """
    Unified downloader

    Args:
        output_dir (str): Target directory for downloaded dataset
        source_type (str): 'kaggle' | 'roboflow'
        params (dict): Source-specific parameters

    Returns:
        str: Path to downloaded dataset
    """
    if source_type == "kaggle":
        return download_kaggle(output_dir, params)
    elif source_type == "roboflow":
        return download_roboflow(output_dir, params)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def download_kaggle(output_dir, params):
    """
    Download dataset from Kaggle

    Args:
        output_dir (str): Target directory
        params (list): [0] dataset anme

    Returns:
        str: Path to downloaded dataset
    """

    dataset_name = params[0]
    if not dataset_name:
        raise ValueError("dataset_name is required for Kaggle downloads")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download dataset
    path = kagglehub.dataset_download(dataset_name)

    # Copy to target folder
    shutil.copytree(path, output_dir, dirs_exist_ok=True)

    # Remove temp folder
    shutil.rmtree(path)

    return output_dir


def download_roboflow(output_dir, params):
    """
    Download dataset from Roboflow

    Args:
        output_dir (str): Target directory
        params (list): e.g. ["object-detection-cfmul", "waste-detection-0momv", 7, "yolov8"]

    Returns:
        str: Path to downloaded dataset
    """
    api_key = "u6dBxE6mzDkitzf9IOgk"
    workspace = params[0]
    project = params[1]
    version = params[2]
    dformat = params[3]

    if not all([api_key, workspace, project, version]):
        raise ValueError(
            "api_key, workspace, project, and version are required for Roboflow downloads"
        )

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)

    version_obj.download(dformat, location=output_dir)

    return output_dir
