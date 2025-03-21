from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO(r"cfg/yolov8_ca.yaml")  # load a pretrained model (recommended for training)
# model = YOLO(r"cfg/yolov8_CBAM.yaml")  # load a pretrained model (recommended for training)
#model = YOLO(r"cfg/yolov8_DAT.yaml")  # load a pretrained model (recommended for training)

# Train the model
model.train(data='./dataset/furniture.yaml', epochs=100, imgsz=640, workers=0)
# model.train(data='./dataset/animal.yaml', epochs=100, imgsz=640, workers=0)
# model.train(data='./dataset/pets.yaml', epochs=100, imgsz=640, workers=0)
