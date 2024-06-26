import cv2
import numpy as np

original_image = None

# Callback function for trackbar
def update_threshold(val):
    global original_image
    threshold1 = cv2.getTrackbarPos('Threshold', 'Canny Edge Detection')
    threshold2 = 2 * threshold1
    thickness = cv2.getTrackbarPos('Thickness', 'Canny Edge Detection')
    edges = cv2.Canny(original_image, threshold1, threshold2)

    # make edges thicker
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=thickness)

    # Invert the image so that the edges are black
    #edges = cv2.bitwise_not(edges)

    cv2.imshow('Canny Edge Detection', edges)


def update_threshold2(val):
    global original_image
    threshold = cv2.getTrackbarPos('Threshold', 'Threshold')
    _, thresholded = cv2.threshold(original_image, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold', thresholded)


def edge_detect_live(image):
    global original_image
    if len(image.shape) == 3:   # If the image is not grayscale, convert it to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a window
    cv2.namedWindow('Canny Edge Detection', cv2.WINDOW_NORMAL)

    # Resize the window to fit the screen
    height = 800
    width = int(image.shape[1] / image.shape[0] * height)
    cv2.resizeWindow('Canny Edge Detection', width, height)

    # Create a trackbar for threshold value and edge thickness
    cv2.createTrackbar('Threshold', 'Canny Edge Detection', 0, 255, update_threshold)
    cv2.createTrackbar('Thickness', 'Canny Edge Detection', 0, 10, update_threshold)

    # Perform initial edge detection
    threshold1 = cv2.getTrackbarPos('Threshold', 'Canny Edge Detection')
    threshold2 = 2 * threshold1
    edges = cv2.Canny(image, threshold1, threshold2)
    edges = cv2.bitwise_not(edges)
    cv2.imshow('Canny Edge Detection', edges)
    original_image = image

    # Wait until the user presses 'Esc' key
    while True:
        key = cv2.waitKey(1)
        try:
            threshold1 = cv2.getTrackbarPos('Threshold', 'Canny Edge Detection')
            thickness = cv2.getTrackbarPos('Thickness', 'Canny Edge Detection')
        except:
            break
        if key == 27:  # 'Esc' key pressed
            break

        if cv2.getWindowProperty('Canny Edge Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    threshold2 = 2 * threshold1

    # Perform edge detection with the updated threshold values
    edges = cv2.Canny(image, threshold1, threshold2)

    # Make edges thicker
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=thickness)

    # Invert the image so that the edges are black
    edges = cv2.bitwise_not(edges)

    # Close all windows
    cv2.destroyAllWindows()

    return edges


def threshold_live(image):
    global original_image
    if len(image.shape) == 3:   # If the image is not grayscale, convert it to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a window
    cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)

    # Resize the window to fit the screen
    height = 800
    width = int(image.shape[1] / image.shape[0] * height)
    cv2.resizeWindow('Threshold', width, height)

    # Create a trackbar for threshold value
    cv2.createTrackbar('Threshold', 'Threshold', 0, 255, update_threshold2)

    # Perform initial edge detection
    threshold = cv2.getTrackbarPos('Threshold', 'Threshold')
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold', thresholded)
    original_image = image

    # Wait until the user presses 'Esc' key
    while True:
        key = cv2.waitKey(1)
        try:
            threshold = cv2.getTrackbarPos('Threshold', 'Threshold')
        except:
            break
        if key == 27:  # 'Esc' key pressed
            break

        if cv2.getWindowProperty('Threshold', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Perform edge detection with the updated threshold values
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Close all windows
    cv2.destroyAllWindows()

    return thresholded

if __name__ == "__main__":
    input_image = "screenshot.png"
    image = cv2.imread(input_image)
    edges = edge_detect_live(image)

    cv2.imshow('Original', image)
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)