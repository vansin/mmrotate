import random
import cv2


import numpy as np


image = cv2.imread("data/icdar2019_tracka_modern_qbox/test_img/cTDaR_t10101.jpg")

height, width, channels = image.shape

center = (width / 2, height / 2)

# angle = random.uniform(-30, 30)
angle = 60

rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

abs_cos = abs(rotation_matrix[0, 0])
abs_sin = abs(rotation_matrix[0, 1])
new_width = int(height * abs_sin + width * abs_cos)
new_height = int(height * abs_cos + width * abs_sin)

rotation_matrix[0, 2] += (new_width / 2) - center[0]
rotation_matrix[1, 2] += (new_height / 2) - center[1]


rotate_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))


# cv2.imshow("rotate_image", rotate_image)

# cv2.waitKey(0)

cv2.imwrite("1.jpg", rotate_image)