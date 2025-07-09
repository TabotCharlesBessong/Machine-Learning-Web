#!/usr/bin/env python3
"""
Fixed training script for football player detection
This script corrects the data.yml vs data.yaml issue
"""

from ultralytics import YOLO
import os

def main():
    # Check if the data.yaml file exists
    data_yaml_path = "football-players-detection-1/data.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} does not exist!")
        print("Available files in football-players-detection-1:")
        if os.path.exists("football-players-detection-1"):
            for file in os.listdir("football-players-detection-1"):
                print(f"  - {file}")
        return
    
    print(f"Found data.yaml at: {data_yaml_path}")
    
    # Load the model
    model = YOLO('yolov5x.pt')
    
    # Train the model with the correct path
    print("Starting training...")
    result = model.train(
        data=data_yaml_path,  # Use the correct .yaml extension
        epochs=10,
        imgsz=640,
        task='detect'
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 