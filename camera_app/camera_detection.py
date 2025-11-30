#!/usr/bin/env python3
"""
Real-time Waste Sorting Detection using Computer Camera
Standalone camera detection application
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# Add parent directory to sys.path to import project modules
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Please install ultralytics first: pip install ultralytics")
    sys.exit(1)

# Color configuration - Material type colors
MATERIAL_COLORS = {
    'METAL': (255, 0, 0),      # Red
    'GLASS': (0, 255, 0),      # Green
    'PLASTIC': (0, 0, 255),    # Blue
    'PAPER': (255, 255, 0),    # Yellow
    'CARDBOARD': (255, 0, 255), # Magenta
}

# Object type colors (lighter shades)
OBJECT_COLORS = {
    'PET_CONTAINER': (100, 100, 255),
    'HDPE_CONTAINER': (120, 120, 255),
    'PLASTIC_WRAPPER': (140, 140, 255),
    'PLASTIC_BAG': (160, 160, 255),
    'TETRAPAK': (255, 100, 255),
    'PAPER_CUP': (150, 150, 0),
    'PAPER_BAG': (170, 170, 0),
    'USED_TISSUE': (190, 190, 0),
    'STYROFOAM': (180, 180, 255),
    'PLASTIC_CUP': (200, 200, 255)
}

class WasteCameraDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        """
        Initialize camera detector
        
        Args:
            model_path: Path to trained .pt model file
            conf_threshold: Detection confidence threshold
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Try other camera indices
            for i in range(1, 4):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Using camera index: {i}")
                    break
            else:
                raise RuntimeError("Cannot open any camera! Please check camera connection")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera initialized successfully")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Confidence threshold: {conf_threshold}")

    def get_color_for_class(self, class_name):
        """Get color based on class name"""
        if class_name in MATERIAL_COLORS:
            return MATERIAL_COLORS[class_name]
        elif class_name in OBJECT_COLORS:
            return OBJECT_COLORS[class_name]
        else:
            # Generate hash-based color for unknown classes
            hash_val = hash(class_name) % (256 * 256 * 256)
            return (hash_val % 256, (hash_val // 256) % 256, (hash_val // 65536) % 256)

    def draw_detections(self, frame, result):
        """Draw detection results on the image"""
        if result.boxes is None or len(result.boxes) == 0:
            return frame, 0
        
        # Get detection boxes, confidence scores and classes
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        detection_count = len(boxes)
        
        # Draw each detection box
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get class name
            class_id = int(cls)
            if hasattr(result, 'names') and result.names and class_id in result.names:
                class_name = result.names[class_id]
            else:
                class_name = f"Class_{class_id}"
            
            # Get color
            color = self.get_color_for_class(class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_text = f"{class_name}: {conf:.2f}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w+10, y1), color, -1)
            
            # Draw label text (white)
            cv2.putText(frame, label_text, (x1+5, y1-5), font, font_scale, (255, 255, 255), thickness)
        
        return frame, detection_count

    def add_info_panel(self, frame, fps, detection_count):
        """Add information panel"""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Information text
        info_texts = [
            f"FPS: {fps:.1f}",
            f"Detections: {detection_count}",
            f"Confidence: {self.conf_threshold:.2f}",
            "",
            "Controls:",
            "Q - Quit program",
            "S - Save screenshot", 
            "+ - Increase confidence",
            "- - Decrease confidence",
            "R - Reset confidence"
        ]
        
        # Create semi-transparent background
        panel_width = 250
        panel_height = len(info_texts) * 25 + 20
        overlay = frame.copy()
        
        # Top-right corner position
        start_x = w - panel_width - 10
        start_y = 10
        
        cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_width, start_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        for i, text in enumerate(info_texts):
            if text:  # Skip empty lines
                y_pos = start_y + 20 + i * 25
                cv2.putText(frame, text, (start_x + 10, y_pos), font, font_scale, color, thickness)
        
        return frame

    def run(self):
        """Run real-time detection"""
        print("\nStarting real-time detection...")
        print("=" * 50)
        
        # FPS calculation variables
        fps_counter = 0
        fps_start_time = cv2.getTickCount()
        fps = 0
        screenshot_counter = 0
        
        try:
            while True:
                # Read camera frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Cannot read camera frame")
                    break
                
                # Execute detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Draw detection results
                frame, detection_count = self.draw_detections(frame, results[0])
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 20:  # Calculate FPS every 20 frames
                    fps_end_time = cv2.getTickCount()
                    fps = 20.0 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                    fps_start_time = fps_end_time
                    fps_counter = 0
                
                # Add information panel
                frame = self.add_info_panel(frame, fps, detection_count)
                
                # Display frame
                cv2.imshow('Waste Sorting Real-time Detection', frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("User exit")
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{screenshot_counter:04d}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
                    screenshot_counter += 1
                elif key == ord('+') or key == ord('='):
                    # Increase confidence
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"Confidence increased to: {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease confidence
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"Confidence decreased to: {self.conf_threshold:.2f}")
                elif key == ord('r') or key == ord('R'):
                    # Reset confidence
                    self.conf_threshold = 0.25
                    print(f"Confidence reset to: {self.conf_threshold:.2f}")
                
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        except Exception as e:
            print(f"\nRuntime error: {e}")
        finally:
            # 清理资源
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resource cleanup complete")

def main():
    parser = argparse.ArgumentParser(
        description="Waste Sorting Real-time Camera Detection App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python camera_detection.py                           # Use default model
  python camera_detection.py --model ../weights/best.pt  # Specify model path
  python camera_detection.py --conf 0.5                # Set confidence threshold
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="../weights/best.pt",
        help="Model file path (default: ../weights/best.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold, range 0-1 (default: 0.25)"
    )
    
    args = parser.parse_args()
    
    # Check model file
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file {model_path} does not exist!")
        print("\nTips:")
        print("1. Please ensure the model file path is correct")
        print("2. If using relative path, ensure running in correct directory")
        print("3. Check if best.pt file exists in weights directory")
        return 1
    
    # Check confidence range
    if not 0 < args.conf <= 1:
        print(f"Error: Confidence threshold {args.conf} out of range (0, 1]")
        return 1
    
    print("Waste Sorting Real-time Detection Starting...")
    print("=" * 50)
    print(f"Model path: {model_path.absolute()}")
    print(f"Confidence threshold: {args.conf}")
    
    try:
        # Initialize detector
        detector = WasteCameraDetector(
            model_path=str(model_path),
            conf_threshold=args.conf
        )
        
        # Run detection
        detector.run()
        
        return 0
        
    except Exception as e:
        print(f"Program startup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)