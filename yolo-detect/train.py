from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO(r"D:\Project\yolo\yolov8-main\ultralytics\cfg\models\v8\yolov8_CA.yaml")  # load a pretrained model (recommended for training)

# Train the model
model.train(data='../dataset/furniture.yaml', epochs=100, imgsz=640, workers=0)
