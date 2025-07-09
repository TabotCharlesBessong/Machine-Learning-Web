# Corrected training cell - Copy this into your notebook
# The issue was using 'data.yml' instead of 'data.yaml'

from ultralytics import YOLO

model = YOLO('yolov5x.pt')

result = model.train(
  data=f'{dataset.location}/data.yaml',  # Changed from data.yml to data.yaml
  epochs=10,
  imgsz=640,
  task='detect'
) 