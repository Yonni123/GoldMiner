import cv2 as cv2
import numpy as np
import random
import string

# Values to be used later
black_border_cont = (None, None, None, None)  # Info about black border
image_height = 720  # Size for every frame, they have to be consistent, the game looks 4:3
image_width = 960


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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top, bottom, left, right = black_border_cont
    if top is None or bottom is None or left is None or right is None:
        # Get the dimensions of the image
        h, w = gray.shape
        # Check for black pixels on the top border
        top = 0
        while top < h and np.all(gray[top] == 0):
            top += 1
        bottom = h - 1  # Check for black pixels on the bottom border
        while bottom >= 0 and np.all(gray[bottom] == 0):
            bottom -= 1
        left = 0  # Check for black pixels on the left border
        while left < w and np.all(gray[:, left] == 0):
            left += 1
        right = w - 1  # Check for black pixels on the right border
        while right >= 0 and np.all(gray[:, right] == 0):
            right -= 1
        black_border_cont = (top, bottom, left, right)
        print(black_border_cont)
    frame = frame[top:bottom + 1, left:right + 1]

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
