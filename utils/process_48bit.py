import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

path = r"E:\DATA\COCO2_65535\images\train2017"
save_path = r"E:\DATA\COCO2\images\train2017\\"
path_list = os.listdir(path)


def process_48bit(img_path, i):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)

    img1 = np.power(img / float(np.max(img)), 1 / 64)
    img1 = np.uint8(cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX))

    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    # cv2.imshow("dst", clahe.apply(img1))
    # cv2.imshow("dst", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(clahe.apply(img1), cmap=plt.cm.gray)
    # plt.pause(1)
    # plt.ioff()
    # cv2.imwrite(save_path+format(i), clahe.apply(img1))

    cv2.imwrite(save_path + format(i), img1)
    print(i)


if __name__ == '__main__':
    for i in path_list:
        # plt.ion()
        img_path = os.path.join(path, i)
        process_48bit(img_path, i)
