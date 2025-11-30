#!/bin/bash

# Waste sorting camera detection startup script

echo "Waste Sorting Real-time Detection Startup Script"
echo "=================================================="

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "Python not installed, please install Python 3.7+ first"
    exit 1
fi

echo "Python version: $(python --version)"

# Check dependencies
echo "Checking dependencies..."
if ! python -c "import cv2" &> /dev/null; then
    echo "OpenCV not installed, installing..."
    pip install opencv-python
fi

if ! python -c "import ultralytics" &> /dev/null; then
    echo "Ultralytics not installed, installing..."
    pip install ultralytics
fi

echo "Dependencies check complete"

# Check model file
MODEL_PATH="../weights/best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file does not exist: $MODEL_PATH"
    echo "Please ensure model file is in correct location"
    exit 1
fi

echo "Model file found: $MODEL_PATH"

# Start detection app
echo "Starting detection app..."
python camera_detection.py --model "$MODEL_PATH" "$@"

echo "Detection app exited"