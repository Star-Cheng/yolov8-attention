import os
import shutil


def image_select():
    iamge_ad = input("请输入选择路径：")
    if iamge_ad == str(0):
        quit()
    tar_ad = input("请输入保存路径：")
    num = 5
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
    print('end')


image_select()
