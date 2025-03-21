import os
import shutil
import cv2
import numpy as np


def augment(augment_path, out_path, augment_mode):
    if augment_mode == 'dark':
        # 读取图片
        image = cv2.imread(augment_path)
        # 低亮度
        dark_image = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
        cv2.imwrite(out_path, dark_image)
    elif augment_mode == 'bright':
        # 读取图片
        image = cv2.imread(augment_path)
        # 高亮度
        bright_image = cv2.convertScaleAbs(image, alpha=2.0, beta=0)
        cv2.imwrite(out_path, bright_image)
    elif augment_mode == 'low':
        # 读取图片
        image = cv2.imread(augment_path)
        # 低对比度
        low_contrast_image = cv2.convertScaleAbs(image, alpha=1.0, beta=50)
        cv2.imwrite(out_path, low_contrast_image)
    elif augment_mode == 'high':
        # 读取图片
        image = cv2.imread(augment_path)
        # 高对比度
        high_contrast_image = cv2.convertScaleAbs(image, alpha=2.0, beta=0)
        cv2.imwrite(out_path, high_contrast_image)
    elif augment_mode == 'noise':
        # 读取图片
        image = cv2.imread(augment_path)
        # 添加高斯噪声
        gaussian_noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, gaussian_noise)
        cv2.imwrite(out_path, noisy_image)
    elif augment_mode == 'blur':
        # 读取图片
        image = cv2.imread(augment_path)
        # 低度锐化
        blur_image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imwrite(out_path, blur_image)
    elif augment_mode == 'sharpening':
        # 读取图片
        image = cv2.imread(augment_path)
        # 高度锐化
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
        cv2.imwrite(out_path, sharpened_image)


image_dir = r'D:\deeplearning\Dataset\Test_data\pets\data\images\train'
label_dir = r'D:\deeplearning\Dataset\Test_data\pets\data\labels\train'
image_augment_dir = r'D:\deeplearning\Dataset\Test_data\pets\data\images\aug'
label_augment_dir = r'D:\deeplearning\Dataset\Test_data\pets\data\labels\aug'
image_listdir = os.listdir(image_dir)
modes = ['dark', 'low']
for image_name in image_listdir[:]:
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, image_name.replace('.' + image_name.split('.')[-1], '.txt'))
    for mode in modes:
        image_augment_path = os.path.join(image_augment_dir, mode + '_' + image_name)
        label_augment_path = os.path.join(label_augment_dir, mode + '_' + image_name.replace('.' + image_name.split('.')[-1], '.txt'))
        # print(image_augment_path, label_augment_path)
        augment(image_path, image_augment_path, mode)
        shutil.copy(label_path, label_augment_path)
print("end")