import os
import shutil


def image_select(imgage_ad, tar_ad, num=1):
    iamge_ad = imgage_ad
    if iamge_ad == str(0):
        quit()
    tar_ad = tar_ad
    num = num
    image_files = os.listdir(iamge_ad)
    selected_im = []
    j = 0
    for i in image_files:
        selected_im.append(i)
        if j < len(selected_im):
            if not os.path.exists(tar_ad):
                os.makedirs(tar_ad)
                shutil.move(os.path.join(iamge_ad, selected_im[j]), tar_ad)
            else:
                shutil.move(os.path.join(iamge_ad, selected_im[j]), tar_ad)
            j += num


data_dir = r'D:\deeplearning\Dataset\Yolo\Tomato pest-diseases.v1-tomato_v1.yolov8'
save_dir = r'D:\deeplearning\Dataset\Test_data\pets'
save_test_dir = os.path.join(save_dir, 'test')
save_test_images_dir = os.path.join(save_test_dir, 'images')
save_test_labels_dir = os.path.join(save_test_dir, 'labels')
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
save_data_dir = os.path.join(save_dir, 'data')
save_data_images_dir = os.path.join(save_data_dir, 'images')
save_data_labels_dir = os.path.join(save_data_dir, 'labels')
save_data_images_train_dir = os.path.join(save_data_images_dir, 'train')
save_data_images_valid_dir = os.path.join(save_data_images_dir, 'val')
save_data_labels_train_dir = os.path.join(save_data_labels_dir, 'train')
save_data_labels_valid_dir = os.path.join(save_data_labels_dir, 'val')
# if not os.path.exists(save_data_dir):
#     os.makedirs(save_data_dir)
train_images_dir = os.path.join(train_dir, 'images')
train_labels_dir = os.path.join(train_dir, 'labels')
valid_images_dir = os.path.join(valid_dir, 'images')
valid_labels_dir = os.path.join(valid_dir, 'labels')
test_images_dir = os.path.join(test_dir, 'images')
test_labels_dir = os.path.join(test_dir, 'labels')
image_select(train_images_dir, save_data_images_train_dir)
image_select(valid_images_dir, save_data_images_valid_dir)
image_select(train_labels_dir, save_data_labels_train_dir)
image_select(valid_labels_dir, save_data_labels_valid_dir)
image_select(test_images_dir, save_test_images_dir)
image_select(test_labels_dir, save_test_labels_dir)
ls = os.listdir(data_dir)
ls.remove('train')
ls.remove('valid')
ls.remove('test')
for file_name in ls:
    file_path = os.path.join(data_dir, file_name)
    save_path = os.path.join(save_dir, file_name)
    print(file_path, save_path)
    shutil.move(file_path, save_path)
