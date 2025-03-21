import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import warnings
import cv2

warnings.filterwarnings('ignore')


def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        # Draw rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Put label text
        label = f'{names[int(cls)]} {conf:.2f}'
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


if __name__ == "__main__":
    model = YOLO(r"C:\Users\admin\PycharmProjects\datasets\models\0061-det-op60.pt")
    img = cv2.imread(r"./dataset/foam.jpg")

    result = model(img, conf=0.5, imgsz=640, half=False)[0]
    boxes = result.boxes.data.tolist()
    names = result.names
    # Draw boxes on the image
    img_with_boxes = draw_boxes(img.copy(), boxes)

    # Save the image with boxes
    output_img_path = r"img_with_boxes.jpg"
    cv2.imwrite(output_img_path, img_with_boxes)

    # Save the boxes to a text file
    output_boxes_path = r"img_boxes.txt"
    with open(output_boxes_path, 'w') as f:
        for box in boxes:
            f.write(f"{box}\n")

    print(f"Image with boxes saved to: {output_img_path}")
    print(f"Boxes saved to: {output_boxes_path}")
