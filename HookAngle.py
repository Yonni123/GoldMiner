import cv2 as cv2
import numpy as np
import Helper
from Helper import display_frame


def get_hook_angle(image):
    background = cv2.imread('Images/HookBackground.png')
    if image.shape != background.shape:
        raise ValueError("Both images must have the same dimensions")
    frame = image.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Remove all pixels that are black in the background image from frame
    frame = np.where(background == 0, 0, frame)

    # Perform background subtraction
    frame = np.where(background == frame, 0, frame)

    display_frame(frame)