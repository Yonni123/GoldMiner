import cv2 as cv2
import numpy as np
import random
import string

# Values to be used later
black_border_cont = None  # In case video feed has black borders, they don't need to be calculated in every frame
image_height = 720  # Size for every frame, they have to be consistent, the game looks 4:3
image_width = 960


# Calulates
def calc_black_border_cont(image):
    # Calculates the size of the black borders, if they exist

    # Parameters:
    #   image (OpenCV frame): The image with or without black border

    # Returns:
    #   black_border_cont(OpenCV Contour): Global contour that can be re-used later to remove black borders
    #   black_border_cont is either a contour if it finds black borders, or -1 if there aren't any
    global black_border_cont

    image = np.array(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for cntrs in contours:
        area = cv2.contourArea(cntrs)
        peri = cv2.arcLength(cntrs, True)
        approx = cv2.approxPolyDP(cntrs, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    cnt = biggest
    if len(cnt) == 0:
        black_border_cont = -1
    else:
        black_border_cont = cv2.boundingRect(cnt)


def preprocess_frame(frame):
    # Removes black borders if it finds any, and resizes the image to image_height and image_width

    # Parameters:
    #   image (OpenCV frame): The image with or without black border

    # Returns:
    #   frame(OpenCV frame): The new preprocessed frame with appropriate size and no black borders
    global black_border_cont, image_height, image_width

    if frame is None:
        raise ValueError("Could not open or find the image.")

    # Remove black borders
    if black_border_cont is None:   # Calculate the contour of the black border
        calc_black_border_cont(frame)

    if black_border_cont != -1:  # -1 means the image has no black borders
        (x, y, w, h) = black_border_cont
        frame = frame[0: y + h, x: x + w]

    # Resize to the specified image_height and image_width
    frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    return frame


def display_frame(frame, ifWait=True):
    # Displays a frame, used to quickly see any frame anywhere in the code

    # Parameters:
    #   image (OpenCV frame): The image to be displayed
    #   ifWait (Boolean): Set to False if you want to display multiple images, and True only in the last one

    # Returns:
    #   NOTHING

    length = 8
    letters = string.ascii_letters
    window_name = ''.join(random.choice(letters) for i in range(length))

    # Display the frame in a window
    print(f"Shape of {window_name}: {frame.shape}")
    cv2.imshow(window_name, frame)
    if ifWait:
        cv2.waitKey(0)
