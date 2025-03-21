import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

color_list = plt.get_cmap("tab20c").colors


if __name__ == "__main__":
    model = YOLO(r"D:\Qianyuan\models\0061-seg0yunmu-241227.pt")
    img_dir = r""
    save_dir = r""
    img = cv2.imread(r"E:\60_img01\2024_12_20_155834_913_01011111.png")
    result = model(img, conf=0.5, imgsz=640, half=False)[0]
    masks = result.masks
    names = result.names
    boxes = result.boxes.data.tolist()
    h, w = img.shape[:2]

    all_mask = np.zeros((h, w)).astype(np.uint8)
    for i, mask in enumerate(masks.data):
        print(f"mask {i}")
        mask = mask.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, (w, h))

        label = int(boxes[i][5])
        color = np.array(color_list[label][:3]) * 255

        colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
        masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)

        mask_indices = mask_resized == 1
        img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)
        all_mask += mask_resized
    cv2.imwrite("result.jpg", img)
    cv2.imwrite("all_mask.jpg", all_mask * 255)
    print("save done")
