

import cv2
import numpy as np

# Load the image
img = cv2.imread("data/ICDAR2019_MTD_HOQ/test_img/cTDaR_t10101.jpg")

# Get the image dimensions
rows, cols = img.shape[:2]

# Define the rotation angle (in degrees)
angle = 60

# Get the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

# Rotate the image
rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

# Show the rotated image
cv2.imwrite("1.jpg", rotated_img)
