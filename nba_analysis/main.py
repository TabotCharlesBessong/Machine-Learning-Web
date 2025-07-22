from ultralytics import YOLO

model = YOLO("models/court/best.pt")  # Load a pretrained YOLOv8 model

# Perform inference on an image
results = model.track("input_videos/video_1.mp4", save=True)
print(results)  # Print the results of the inference
print("======================")
for box in results[0].boxes:
  print(f"Box: {box.xyxy}")