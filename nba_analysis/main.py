from ultralytics import YOLO

model = YOLO("yolov8x")  # Load a pretrained YOLOv8 model

# Perform inference on an image
results = model.predict("input_videos/video_3.mp4", save=True)
print(results)  # Print the results of the inference
print("======================")
for box in results[0].boxes:
  print(f"Box: {box.xyxy}")