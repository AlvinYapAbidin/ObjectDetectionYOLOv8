from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml") # build  a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=1) # train the model
results = model.val()
results = model.export(format="onnx", imgsz=(384,640), opset=12)
