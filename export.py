import os
from ultralytics import YOLO


model_path = os.path.join('.', 'runs','detect', 'train4', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)

model.export(format="onnx", opset_version=12)