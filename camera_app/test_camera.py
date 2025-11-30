#!/usr/bin/env python3
"""
Simple camera test script
"""

import cv2
import sys

def test_camera():
    print("Testing camera access...")
    
    # Try different camera indices
    for i in range(4):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"SUCCESS: Camera {i} is working!")
                print(f"Frame shape: {frame.shape}")
                cap.release()
                return i
            else:
                print(f"Camera {i} opened but cannot read frames")
        else:
            print(f"Camera {i} failed to open")
        
        cap.release()
    
    print("ERROR: No working camera found!")
    return -1

def check_permissions():
    print("Checking camera permissions...")
    print("If you see permission dialogs, please click 'Allow'")
    
    # This should trigger permission request
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("Camera permission granted!")
            print(f"Frame captured: {frame.shape}")
        else:
            print("Camera opened but no frame captured")
    else:
        print("Camera permission denied or camera not available")
    
    cap.release()

if __name__ == "__main__":
    print("Camera Permission Test")
    print("=" * 30)
    
    check_permissions()
    print()
    camera_index = test_camera()
    
    if camera_index >= 0:
        print(f"\nCamera test successful! Use camera index: {camera_index}")
        print("You can now run the main detection app.")
    else:
        print("\nCamera test failed!")
        print("Try:")
        print("1. Grant camera permission in System Preferences")
        print("2. Install opencv-python-headless instead")
        print("3. Check if camera is being used by other apps")