#!/bin/bash

# Data Processing Script for Waste Sorting Project
# Usage: ./run.sh [command] [options]

set -e  # Exit on any error

DATA_PROCESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$DATA_PROCESS_DIR/generate.py"
REQUIREMENTS="$DATA_PROCESS_DIR/requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Install dependencies if requirements.txt exists
install_deps() {
    if [ -f "$REQUIREMENTS" ]; then
        print_status "Installing dependencies..."
        pip install -r "$REQUIREMENTS"
    else
        print_warning "Requirements file not found: $REQUIREMENTS"
    fi
}

# Show usage information
show_help() {
    echo "Data Processing Script for Waste Sorting Project"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup           Install dependencies from requirements.txt"
    echo "  download        Download all datasets"
    echo "  transform       Transform datasets to YOLO format"
    echo "  merge           Merge transformed datasets"
    echo "  all             Run complete pipeline (download + transform + merge)"
    echo "  test-draw       Draw labels on image (requires --image-path and --label-path)"
    echo "  test-count      Count images in directory (requires --image-dir)"
    echo "  test-dist       Show class distribution (requires --merge-dir)"
    echo "  help            Show this help message"
    echo ""
    echo "Options:"
    echo "  --config FILE   Path to configuration YAML file"
    echo "  --image-path    Path to image file (for test-draw)"
    echo "  --label-path    Path to label file (for test-draw)"
    echo "  --image-dir     Path to images directory (for test-count)"
    echo "  --merge-dir     Path to merged dataset directory (for test-dist)"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 all"
    echo "  $0 download --config my-config.yaml"
    echo "  $0 test-draw --image-path img.jpg --label-path label.txt"
    echo "  $0 test-count --image-dir ./images"
    echo "  $0 test-dist --merge-dir ./merged_dataset"
}

# Main script logic
case "${1:-help}" in
    setup)
        install_deps
        ;;
    download)
        print_status "Starting dataset download..."
        cd "$DATA_PROCESS_DIR" && python main.py download "${@:2}"
        ;;
    transform)
        print_status "Starting dataset transformation..."
        cd "$DATA_PROCESS_DIR" && python main.py transform "${@:2}"
        ;;
    merge)
        print_status "Starting dataset merge..."
        cd "$DATA_PROCESS_DIR" && python main.py merge "${@:2}"
        ;;
    all)
        print_status "Starting complete data processing pipeline..."
        cd "$DATA_PROCESS_DIR" && python main.py all "${@:2}"
        ;;
    test-draw)
        print_status "Testing label drawing..."
        cd "$DATA_PROCESS_DIR" && python main.py test --func draw "${@:2}"
        ;;
    test-count)
        print_status "Counting images..."
        cd "$DATA_PROCESS_DIR" && python main.py test --func count "${@:2}"
        ;;
    test-dist)
        print_status "Showing class distribution..."
        cd "$DATA_PROCESS_DIR" && python main.py test --func distribution "${@:2}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac