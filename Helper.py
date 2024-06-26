import cv2 as cv2
import numpy as np
import random
import string
import Constants

# Values to be used later
black_border_cont = (None, None, None, None)  # Info about black border

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


def preprocess_frame(frame):
    # Removes black borders if it finds any, and resizes the image to image_height and image_width

    # Parameters:
    #   image (OpenCV frame): The image with or without black border

    # Returns:
    #   frame(OpenCV frame): The new preprocessed frame with appropriate size and no black borders
    global black_border_cont

    if frame is None:
        raise ValueError("Could not open or find the image.")

    # Remove black borders
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top, bottom, left, right = black_border_cont
    if top is None or bottom is None or left is None or right is None:
        # Get the dimensions of the image
        h, w = gray.shape
        threshold = 10  # Threshold since complete "black" pixels might be bugged out by noise
        # Check for black pixels on the top border
        top = 0
        while top < h and np.mean(gray[top]) < threshold:
            top += 1
        bottom = h - 1  # Check for black pixels on the bottom border
        while bottom >= 0 and np.mean(gray[bottom]) < threshold:
            bottom -= 1
        left = 0  # Check for black pixels on the left border
        while left < w and np.mean(gray[:, left]) < threshold:
            left += 1
        left += 2
        right = w - 1  # Check for black pixels on the right border
        while right >= 0 and np.mean(gray[:, right]) < threshold:
            right -= 1
        right -= 2
        black_border_cont = (top, bottom, left, right)
        print(black_border_cont)
    frame = frame[top:bottom + 1, left:right + 1]

    # Resize to the specified image_height and image_width
    frame = cv2.resize(frame, (Constants.image_width, Constants.image_height), interpolation=cv2.INTER_AREA)
    return frame
