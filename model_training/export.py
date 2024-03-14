import os
from ultralytics import YOLO


model_path = os.path.join('.', 'runs','detect', 'train4', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)

# When in venv to use python ver11, run through command line
model.export(format="onnx", opset=12)