#!/usr/bin/env python3
"""
Comprehensive solution for the football training issue
This script fixes the data.yml vs data.yaml problem and path issues
"""

import os
import shutil
from ultralytics import YOLO

def fix_data_yaml():
    """Fix the data.yaml file with correct paths"""
    original_path = "football-players-detection-1/data.yaml"
    backup_path = "football-players-detection-1/data.yaml.backup"
    
    # Create backup of original file
    if os.path.exists(original_path):
        shutil.copy2(original_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Create corrected data.yaml
    corrected_content = """names:
- ball
- goalkeeper
- player
- referee
nc: 4
roboflow:
  license: CC BY 4.0
  project: football-players-detection-3zvbc
  url: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1
  version: 1
  workspace: roboflow-jvuqo
test: football-players-detection-1/test/images
train: football-players-detection-1/train/images
val: football-players-detection-1/valid/images
"""
    
    with open(original_path, 'w') as f:
        f.write(corrected_content)
    
    print(f"Fixed {original_path} with correct paths")

def verify_dataset_structure():
    """Verify that the dataset structure is correct"""
    required_dirs = [
        "football-players-detection-1/train/images",
        "football-players-detection-1/train/labels", 
        "football-players-detection-1/valid/images",
        "football-players-detection-1/valid/labels",
        "football-players-detection-1/test/images",
        "football-players-detection-1/test/labels"
    ]
    
    print("Verifying dataset structure...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"✓ {dir_path}: {file_count} files")
        else:
            print(f"✗ {dir_path}: MISSING")
            return False
    
    return True

def train_model():
    """Train the YOLO model with correct configuration"""
    data_yaml_path = "football-players-detection-1/data.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} does not exist!")
        return
    
    print(f"Using data configuration: {data_yaml_path}")
    
    # Load the model
    model = YOLO('yolov5x.pt')
    
    # Train the model
    print("Starting training...")
    result = model.train(
        data=data_yaml_path,
        epochs=10,
        imgsz=640,
        task='detect'
    )
    
    print("Training completed!")
    return result

def main():
    print("=== Football Player Detection Training Fix ===")
    print()
    
    # Step 1: Fix the data.yaml file
    print("Step 1: Fixing data.yaml file...")
    fix_data_yaml()
    print()
    
    # Step 2: Verify dataset structure
    print("Step 2: Verifying dataset structure...")
    if not verify_dataset_structure():
        print("Error: Dataset structure is incorrect!")
        return
    print()
    
    # Step 3: Train the model
    print("Step 3: Training the model...")
    train_model()
    print()
    
    print("=== All done! ===")

if __name__ == "__main__":
    main() 