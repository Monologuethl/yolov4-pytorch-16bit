'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image

from yolo import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
yolo = YOLO()

# test_path = r"./VOCdevkit-1/tets_heatmap"
# test_path = r"./VOCdevkit/tets_heatmap"
test_path=r"/home/techik/Project/DATASET/VOCdevkit-xilanhua/VOC2007/JPEGImages"
test_path_list = os.listdir(test_path)
plt.ion()

for i in test_path_list:
    img = os.path.join(test_path, i)
    # img = input('Input image filename:')
    try:
        image = Image.open(img)
        # image = cv2.imread(img, -1)
    except:
        print('Open Error! Try again!')
        continue
    else:
        # r_image = yolo.detect_16bit_image(image)
        r_image = yolo.detect_image(image)

        # img1 = np.power(r_image / float(np.max(r_image)), 1 / 64)
        # img1 = np.uint8(cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX))
        cv2.imshow("res", r_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # plt.imshow(r_image/np.max(r_image))
        # plt.pause(1)
        # plt.ioff()
