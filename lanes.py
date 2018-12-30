import cv2
import numpy as np

# Read the image
image = cv2.imread('test_image.jpg')
# Make a copy of the image to no affect the original
lane_image = np.copy(image)
# Use the copy of the image as a base to generate a gray scale one
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# Applying a gaussian 5,5 matrix to normalize the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Uses a derivatice to measures the adjacent changes in intensity
# in all directions of the images, which returns the gradient
canny = cv2.Canny(blur, 50, 150)

# Show the image
cv2.imshow('result', canny)
# Forever
cv2.waitKey(0)
