import cv2
import numpy as np


def cany(image):
    # Use the copy of the image as a base to generate a gray scale one
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Applying a gaussian 5,5 matrix to normalize the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Uses a derivatice to measures the adjacent changes in intensity
    # in all directions of the images, which returns the gradient
    return cv2.Canny(blur, 50, 150)


# Mark the region of interest, in this case the line
def region_of_interest(image):
    # Height of the image (y-axis), image.shape returns [y,x]
    height = image.shape[0]

    # Cretes an list of polygons, which are formed by a list
    # of it's cartecian points, one poly -> [(x1,y2)...(xn,yn)]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    # Creates a matrix the same size of the original image
    mask = np.zeros_like(image)
    # Create a polygon (shape) in the mask matrix, white filled
    cv2.fillPoly(mask, polygons, 255)

    return mask


# Read the image
image = cv2.imread('test_image.jpg')
# Make a copy of the image to no affect the original
lane_image = np.copy(cany(image))

# Show the image
cv2.imshow('result', region_of_interest(lane_image))
# Forever
cv2.waitKey(0)
