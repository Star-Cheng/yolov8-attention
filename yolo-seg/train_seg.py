from ultralytics import YOLO

# Load a model
model = YOLO(r"yolov8n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO(r"C:\Users\admin\PycharmProjects\Yolo\ultralytics\ultralytics\cfg\models\v8\yolov8.yaml")  # load a pretrained model (recommended for training)

# Train the model
model.train(data='./dataset/yunmu.yaml', epochs=100, batch=8, imgsz=640, workers=0, device="cuda:0")
