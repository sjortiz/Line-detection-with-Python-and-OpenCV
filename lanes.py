import cv2
import numpy as np


def make_coordinates(image: list, line_parameters: list) -> list:
    """ Generates a list with the Ys and Xs values 
    of the line to be draw
    """

    # unpack the slope and the intercept
    slope, intercept = line_parameters
    # y1 = image height which is the lower conner
    y1 = image.shape[0]
    # calculate y2, line ending as 3/5 of y1 to set up
    y2 = int(y1 * (3/5))
    # <<x = (y - b)>> / slope; clearence of <<y = mx + b>>
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    # return array with the two points that form the line
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image: list, lines: list) -> list:
    """Calculates the average slope intercept to draw a common line
    that can be seen as the line of best fit
    """

    # Initialize list to save the left and right lines
    left_fit = []
    right_fit = []

    for line in lines:
        # converts [[x1, y1, x2, y2]] to [x1, y1, x2, y2]
        x1, y1, x2, y2 = line.reshape(4)
        # Generastes a polygon based on the specified pounts
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # Takes the slope (/) and the intercept (-)
        slope = parameters[0]
        intercept = parameters[1]

        # if slope is negative, the line is on the left side
        if slope < 0:
            left_fit.append((slope, intercept))

        # if slope is positive, the line is on the left side
        else:
            right_fit.append((slope, intercept))

    # takes the average on the x axis only (the slope)
    # To make a consistent line instead of multiple lines
    # with short divition inbetween
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    # calculate X to draw the lines
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)

    # return the lines [[coordenates], [coordenates]]
    return np.array([left_line, right_line])


def canny(image: list) -> object:
    """Converts the image to gray scalle, normalize the pixels
    and apply a canny derivative to return an image with just
    the lines in black and white, where only whites are the pixels
    that where above 150 of 255 in their values
    """

    # Use the copy of the image as a base to generate a gray scale one
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Applying a gaussian 5,5 matrix to normalize the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Uses a derivatice to measures the adjacent changes in intensity
    # in all directions of the images, which returns the gradient
    return cv2.Canny(blur, 50, 150)


def display_lines(image: object, lines: list):
    """ Adds the lines to a black image of the same
    size of the one being processed
    """

    # Generastes a black image of the same size
    line_image = np.zeros_like(image)

    # if here is any line
    if lines.any():
        # for each line
        for x1, y1, x2, y2 in lines:
            # Draw the line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    # return the image with the lines
    return line_image


def region_of_interest(image: object) -> object:
    """Mark the region of interest, in this case the line
    """

    # Height of the image (y-axis), image.shape returns [y,x]
    height = image.shape[0]

    # Creates an list of polygons, which are formed by a list
    # of it's cartecian points, one poly -> [(x1,y2)...(xn,yn)]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    # Creates a matrix the same size of the original image
    mask = np.zeros_like(image)
    # Create a polygon (shape) in the mask matrix, white filled
    cv2.fillPoly(mask, polygons, 255)

    return cv2.bitwise_and(image, mask)


def process(frame: list) -> object:
    """Function that calls the necessary transformations
    over the image to display the lines over it
    """

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(
        cropped_image, 2, np.pi/180, 100, np.array([]),
        minLineLength=40, maxLineGap=5)

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def read_entity(name: str, video: bool = False) -> None:
    """Read the file, depeding if it's image or video
    """

    if video:

        cap = cv2.VideoCapture(name)

        while(True):

            _, frame = cap.read()

            # Show the image
            cv2.imshow('result', process(frame))

            # wait 1 ms
            if cv2.waitKey(1) == ord('q'):
                return

        cv2.destroyAllWindows()
        cap.release()

    cap = cv2.imread(name)
    cv2.imshow('result', process(cap))
    cv2.waitKey(0)


if __name__ == '__main__':
    read_entity("test2.mp4", video=True)
