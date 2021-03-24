import cv2 as cv
import numpy as np

# 载入手写数字图片
img = cv.imread('handwriting.jpg', 0)
# 将图像二值化
_, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(thresh, 3, 2)

# 创建出两幅彩色图用于绘制
img_color1 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
img_color2 = np.copy(img_color1)
# 创建一幅彩色图像用作结果对比
img_color = np.copy(img_color1)

# 计算数字1的轮廓特征
cnt = contours[1]
cv.drawContours(img_color1, [cnt], 0, (0, 0, 255), 2)

# 1.轮廓面积
area = cv.contourArea(cnt)  # 6289.5
print(area)

# 2.轮廓周长
perimeter = cv.arcLength(cnt, True)  # 527.4041
print(perimeter)

# 3.图像矩
M = cv.moments(cnt)
print(M)
print(M['m00'])  # 轮廓面积
cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']  # 轮廓质心
print(cx, cy)

# 4.图像外接矩形和最小外接矩形
x, y, w, h = cv.boundingRect(cnt)  # 外接矩形
cv.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)

rect = cv.minAreaRect(cnt)  # 最小外接矩形
box = np.int0(cv.boxPoints(rect))  # 矩形的四个角点并取整
cv.drawContours(img_color1, [box], 0, (255, 0, 0), 2)

# 5.最小外接圆
(x, y), radius = cv.minEnclosingCircle(cnt)
(x, y, radius) = map(int, (x, y, radius))  # 这也是取整的一种方式噢
cv.circle(img_color2, (x, y), radius, (0, 0, 255), 2)

# 6.拟合椭圆
ellipse = cv.fitEllipse(cnt)
cv.ellipse(img_color2, ellipse, (0, 255, 0), 2)

result = np.hstack((img_color1,img_color2))
cv.imshow('result',result)
cv.waitKey(0)
cv.destroyAllWindows()
