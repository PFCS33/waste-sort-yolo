# Waste Sorting Real-time Camera Detection App

A real-time waste sorting detection application based on YOLO models, using computer camera for live detection and classification.

## Features

- **Real-time Detection**: Live waste detection using computer camera
- **Multi-category Support**: Supports various waste material and object types
- **Adjustable Parameters**: Real-time confidence threshold adjustment
- **Screenshot Capture**: One-click screenshot saving of detection results
- **FPS Display**: Real-time frame rate display
- **Color-coded Labels**: Different categories use different colors for distinction

## Quick Start

### 1. Install Dependencies

```bash
# In camera_app directory
pip install -r requirements.txt
```

### 2. Check Model File

Ensure you have the trained model file `best.pt` in the parent directory's `weights/` folder:

```
waste-sorting/
├── camera_app/
│   ├── camera_detection.py
│   ├── requirements.txt
│   └── README.md
└── weights/
    └── best.pt  # Your trained model
```

### 3. Run Detection App

```bash
# Use default settings
python camera_detection.py

# Specify model path
python camera_detection.py --model ../weights/best.pt

# Set confidence threshold
python camera_detection.py --conf 0.5
```

## Controls

| Key | Function |
|-----|----------|
| `Q` | Quit program |
| `S` | Save current frame screenshot |
| `+` or `=` | Increase confidence threshold (+0.05) |
| `-` | Decrease confidence threshold (-0.05) |
| `R` | Reset confidence threshold to 0.25 |

## Interface Description

- **Detection Boxes**: Different colored bounding boxes identify different waste categories
- **Labels**: Display category name and confidence score
- **Info Panel**: Top-right corner shows FPS, detection count, etc.
- **Real-time Adjustment**: Live parameter adjustment during detection

## Supported Waste Categories

### Material Categories
- **METAL** - Red
- **GLASS** - Green  
- **PLASTIC** - Blue
- **PAPER** - Yellow
- **CARDBOARD** - Magenta

### Object Categories
- PET containers, HDPE containers, plastic wrapping, plastic bags
- Tetra Pak, paper cups, paper bags, used tissues
- Styrofoam, plastic cups, etc.

## Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--model` | Model file path | `../weights/best.pt` | Valid .pt file path |
| `--conf` | Confidence threshold | `0.25` | 0.01 - 1.0 |

## Troubleshooting

### Camera Cannot Open
1. Check if camera is being used by other applications
2. Try closing other video call software
3. On Mac, ensure camera permissions: System Preferences → Security & Privacy → Camera

### Model Loading Failed
1. Ensure model file path is correct
2. Check if model file is complete
3. Ensure correct version of ultralytics is installed

### Slow Detection Speed
1. Lower camera resolution
2. Appropriately increase confidence threshold
3. Ensure sufficient computational resources

### Dependency Installation Issues
```bash
# If opencv installation has issues, try:
pip uninstall opencv-python
pip install opencv-python-headless

# If torch installation has issues, visit: https://pytorch.org/get-started/locally/
```

## Output Files

- **Screenshot files**: `screenshot_XXXX.jpg` - Saved in running directory
- **Detection logs**: Real-time console output

## Tech Stack

- **OpenCV**: Camera capture and image processing
- **YOLO (Ultralytics)**: Object detection model
- **NumPy**: Numerical computing
- **Python 3.7+**: Programming language

## Usage Tips

1. **Lighting**: Ensure good lighting conditions
2. **Camera Angle**: Keep camera stable with clear object visibility
3. **Detection Distance**: Maintain appropriate detection distance, avoid objects being too small or large
4. **Background**: Use clean background to avoid interference

## Update Log

- **v1.0**: Basic real-time detection functionality
- Support for multiple waste categories
- Real-time parameter adjustment
- Screenshot saving functionality

---

**Issue Reports**: If you encounter problems, please check console output or contact the developer.