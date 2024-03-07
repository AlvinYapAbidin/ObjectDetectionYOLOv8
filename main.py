from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml") # build  a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=100) # train the model
#results = model.val()
#results = model.export(format="onnx")
