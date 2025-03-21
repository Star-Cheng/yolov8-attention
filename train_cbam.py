from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO(r".\ultralytics\cfg\models\v8\yolov8.yaml")  # build a new model from YAML
# model = YOLO(r"cfg/yolov8_ca.yaml")  # load a pretrained model (recommended for training)
# model = YOLO(r"cfg/yolov8_CBAM.yaml")  # load a pretrained model (recommended for training)
#model = YOLO(r"cfg/yolov8_DAT.yaml")  # load a pretrained model (recommended for training)

# Train the model
model.train(data='./dataset/building.yaml', epochs=50, imgsz=640, batch=50, workers=0, device=0)
