# BinGo: Multi-Item Waste Classification for Intelligent Sorting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hierarchical object detection system for real-time waste sorting, featuring multi-label classification with material-level fallback for robust handling of novel objects.


<p align="center">
  <img src="assets/pipeline.png" alt="BinGo System Pipeline" width="800"/>
</p>

## Highlights

- **Multi-Item Detection**: Identifies multiple waste items simultaneously using YOLOv8
- **Hierarchical Classification**: Two-level taxonomy (5 materials → 14 objects) with fallback mechanism
- **114% mAP Improvement**: From 0.36 (baseline) to 0.77 through dataset optimization
- **Mobile Ready**: TensorFlow Lite deployment for real-time Android inference

## Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/8a209090-e66f-4c9f-a526-24c9edacc63b" alt="BinGo Demo" width="300"/>
</p>


## Architecture

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
python predict.py --source path/to/image.jpg --weights weights/best.pt
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

```bash
# Configure datasets in configs/dataset.yaml, then run:
python scripts/prepare_dataset.py --config configs/dataset.yaml
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
python train.py --data configs/data.yaml --cfg yolov8n.yaml --epochs 300

# Hierarchical multi-label training
python train_hierarchical.py --data configs/data.yaml --loss bce_penalty --epochs 300
```

<details>
<summary><b>Key Arguments</b></summary>

- `--loss`: Loss function (`bce` | `bce_penalty`)
- `--imgsz`: Input image size (default: 640)
- `--batch`: Batch size (default: 16)

</details>

## Results

### Performance Comparison

| Stage | Approach | mAP@0.5:0.95 | Novel Object Acc |
|-------|----------|--------------|------------------|
| 1 | Baseline (7-class) | 0.36 | 23% |
| 2 | Fine-grained (14-class) | 0.77 | - |
| 3 | Hierarchical + Fallback | 0.76 | 64% |

### Qualitative Results

<p align="center">
  <img src="assets/comparison.png" alt="Detection Comparison" width="700"/>
</p>

*Left: Baseline misses paper bag and mislabels HDPE container. Right: Our model correctly detects both.*

## Project Structure

```
waste-sort-yolo/
├── configs/           # Dataset and training configs
├── scripts/           # Data preparation pipeline
├── src/
│   ├── dataset/       # Custom YOLODataset for multi-hot labels
│   ├── loss/          # BCE and hierarchical penalty loss
│   ├── models/        # Modified YOLOv8 architecture
│   └── utils/         # NMS with fallback, evaluation
├── weights/           # Pretrained models
├── train.py
├── train_hierarchical.py
└── predict.py
```

## Related Repositories

| Repository | Description |
|------------|-------------|
| [panqier/vcproject](https://github.com/panqier/vcproject) | Android app with TFLite deployment |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8 base implementation |

## Citation

```bibtex
@misc{bingo2025,
  author = {Huiyou Liu and Qier Pan and Yunshan Feng},
  title = {BinGo: Multi-Item Waste Classification for Intelligent Sorting},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/PFCS33/waste-sort-yolo}
}
```

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 implementation
- [TACO](http://tacodataset.org/), [TrashNet](https://github.com/garythung/trashnet), and other public datasets
- Course project for CMPT 742 Visual Computing at Simon Fraser University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
