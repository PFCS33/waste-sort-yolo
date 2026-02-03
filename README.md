# BinGo: Multi-Item Waste Classification for Intelligent Sorting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hierarchical object detection app for real-time waste sorting, featuring multi-label classification with material-level prediction fallback for robust handling of novel objects.

## Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/8a209090-e66f-4c9f-a526-24c9edacc63b" alt="BinGo Demo" width="300"/>
</p>

## Highlights

- **Multi-Item Detection**: Identifies multiple waste items simultaneously using YOLOv8
- **Hierarchical Classification**: Two-level taxonomy (5 materials → 14 objects) with fallback mechanism
- **114% mAP Improvement**: From 0.36 (baseline) to 0.77 through dataset optimization
- **Mobile Ready**: TensorFlow Lite deployment for real-time Android inference



## Workflow

```
                    ┌─────────────────┐
                    │   5 Materials   │
                    │ Metal│Glass│... │
                    └────────┬────────┘
                             │ fallback when uncertain
┌──────────┐    ┌───────┐    ▼
│  Image   │───>│ YOLO  │───> 14 Object Types ───> Regional Bin Mapping
└──────────┘    └───────┘    PET│HDPE│Wrapper...    (Garbage/Recycle/Mixed)
```

## Quick Start

### Installation

```bash
git clone https://github.com/PFCS33/waste-sort-yolo.git
cd waste-sort-yolo

# Create environment
conda create -n bingo python=3.10 -y
conda activate bingo

# Install dependencies
pip install -r requirements.txt
```

### Inference

```bash
python main.py --source path/to/image.jpg --weights weights/best.pt
```

## Dataset

We constructed a balanced 14-class dataset by merging 13+ public sources.

| Metric | Baseline Dataset | Our Dataset |
|--------|------------------|-------------|
| Classes | 7 (materials) | 14 (objects) |
| Imbalance Ratio | >50 | 4.84 |
| Images | ~5k | ~37k |

<details>
<summary><b>Class Taxonomy</b></summary>

| Material | Object Types |
|----------|-------------|
| Plastic | PET Container, HDPE Container, Plastic Wrapper, Plastic Bag, Plastic Cup, Styrofoam, Tetra Pak |
| Paper | Paper, Paper Cup, Paper Bag, Used Tissue |
| Metal | Metal |
| Glass | Glass |
| Cardboard | Cardboard |

</details>


### Data Preparation

Our automated pipeline downloads, transforms, and merges multiple public datasets using a YAML configuration.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6ecb0f22-9f56-4b5a-99a2-57666e782305" alt="Data Pipeline" width="450"/>
</p>

**YAML Configuration Template:**

```yaml
global:
  root_dir: "data_merge"          # Output directory
  target_format: "yolo"           # Annotation format
  nc: 14                          # Number of classes
  split: [0.8, 0.1, 0.1]          # Train/val/test split ratio
  classes:                        # Unified class names
    - METAL
    - GLASS
    - PET_CONTAINER
    # ... (14 classes total)

datasets:
  - name: "dataset-name"
    download:
      source: "roboflow"          # Source: roboflow | kaggle
      params: ["workspace", "project", version, "yolov8"]
    transform:
      exclude_class: [0, 2]       # Classes to exclude (optional)
      class_mapping:              # Map source labels to unified labels
        0: 8                      # source_id: target_id
        1: 1
    paths:                        # Image/label paths in source dataset
      - images: ["train", "images"]
        labels: ["train", "labels"]
```

**Run the pipeline:**

```bash
bash run_data.sh
```

## Model Zoo

| Model | mAP@0.5 | mAP@0.5:0.95 | Params | Download |
|-------|---------|--------------|--------|----------|
| Baseline (7-class) | 0.54 | 0.36 | 3.2M | - |
| Improved (14-class) | 0.92 | 0.77 | 3.2M | [weights](weights/) |
| Hierarchical (BCE) | 0.89 | 0.75 | 3.2M | [weights](weights/) |
| Hierarchical (BCE+penalty) | 0.90 | 0.76 | 3.2M | [weights](weights/) |

## Training

```bash
# Standard training
python main.py --mode train --data configs/data.yaml --epochs 300

# Hierarchical multi-label training
python main.py --mode train --loss bce_penalty --epochs 300
```


## Results

### Performance Comparison

| Stage | Approach | mAP@0.5:0.95 | Novel Object Acc |
|-------|----------|--------------|------------------|
| 1 | Baseline (7-class) | 0.36 | 23% |
| 2 | Fine-grained (14-class) | 0.77 | - |
| 3 | Hierarchical + Fallback | 0.76 | 64% |


## Project Structure

```
waste-sort-yolo/
├── main.py                 # Entry point for training/inference
├── run_data.sh             # Dataset preparation script
├── weights/                # Pretrained model weights
└── scripts/
    ├── data/               # Dataset pipeline
    │   ├── config.yaml     # Dataset configuration
    │   ├── config_h.yaml   # Hierarchical config
    │   └── utils/          # Download, transform, merge utilities
    ├── models/
    │   ├── baseline/       # Standard YOLOv8 training
    │   └── multi_label/    # Hierarchical multi-label YOLO
    │       ├── config.py   # Hierarchy configuration
    │       ├── dataset.py  # Multi-hot label generation
    │       ├── loss.py     # BCE & hierarchical penalty loss
    │       ├── nms.py      # NMS with fallback mechanism
    │       ├── predictor.py
    │       └── trainer.py
    └── utils.py
```

## Related Repositories

| Repository | Description |
|------------|-------------|
| [panqier/vcproject](https://github.com/panqier/vcproject) | Android app with TFLite deployment |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8 base implementation |



## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [TACO](http://tacodataset.org/), [TrashNet](https://github.com/garythung/trashnet), and other public datasets
- Course project for CMPT 742 Visual Computing at Simon Fraser University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
