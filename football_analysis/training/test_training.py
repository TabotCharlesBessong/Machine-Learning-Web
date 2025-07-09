#!/usr/bin/env python3
"""
Test script to verify the training configuration works
"""

from ultralytics import YOLO
import os

def test_training():
    """Test the training configuration"""
    
    # Check if data.yaml exists
    data_yaml_path = "football-players-detection-1/data.yaml"
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} does not exist!")
        return False
    
    print(f"✓ Found data.yaml at: {data_yaml_path}")
    
    # Check if the paths in data.yaml are correct
    with open(data_yaml_path, 'r') as f:
        content = f.read()
        print("Data.yaml content:")
        print(content)
    
    # Check if the directories exist (corrected paths)
    directories_to_check = [
        "football-players-detection-1/football-players-detection-1/train/images",
        "football-players-detection-1/football-players-detection-1/train/labels",
        "football-players-detection-1/football-players-detection-1/valid/images", 
        "football-players-detection-1/football-players-detection-1/valid/labels",
        "football-players-detection-1/football-players-detection-1/test/images",
        "football-players-detection-1/football-players-detection-1/test/labels"
    ]
    
    print("\nChecking directories:")
    for dir_path in directories_to_check:
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"✓ {dir_path}: {file_count} files")
        else:
            print(f"✗ {dir_path}: MISSING")
            return False
    
    print("\n✓ All directories exist!")
    print("✓ Configuration is ready for training!")
    
    return True

if __name__ == "__main__":
    test_training() 