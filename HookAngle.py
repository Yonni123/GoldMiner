import cv2 as cv2
import numpy as np
from Helper import display_frame
from ObjectDetection import get_bounding_boxes
import Constants


def overlay_protractor(frame):
    # Read the overlay image with alpha channel
    overlay_img = cv2.imread("Images/FinalAngle2.png", cv2.IMREAD_UNCHANGED)

    # Check if the overlay image has an alpha channel
    if overlay_img.shape[2] != 4:
        raise ValueError("Protractor image must have 4 channels (including alpha channel)")

    # Split the overlay image into its color and alpha channels
    overlay_img_color = overlay_img[:, :, :3]
    overlay_img_alpha = overlay_img[:, :, 3] / 255.0

    # Blend the overlay image and the background image
    for c in range(3):  # Loop over the color channels
        frame[:, :, c] = (overlay_img_alpha * overlay_img_color[:, :, c] +
                          (1.0 - overlay_img_alpha) * frame[:, :, c])

    return frame


def get_largest_cont(conts):
    largest_cont = None
    largest_area = 0
    for i, cnt in enumerate(conts):
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_cont = cnt
    return largest_cont


def extract_wire_mask(frame):  # Extract a cropped frame of where the wire attached to the hook is (the one swinging)
    # Extract the bounding box of the blue circle behind the miner
    lower = np.array([106, 152, 119])  # (hMin = 106 , sMin = 152, vMin = 119)
    upper = np.array([125, 255, 255])  # (hMax = 125 , sMax = 255, vMax = 255)
    vhs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(vhs, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours

    # Make sure we actually found something meaningful (otherwise it will crash during loading screen for example)
    if len(contours) == 0:
        return -1
    largest_cont = get_largest_cont(contours)
    if largest_cont is None or len(largest_cont) == 0:
        return -1
    if cv2.contourArea(largest_cont) < 100:
        return -1
    bb = get_bounding_boxes([largest_cont])[0]

    # Crop the image further down to only allow the swinging wire in the frame based on some constants
    (x, y, w, h) = bb
    cropped_image = frame[y:y + h, x:x + w]
    y_s = int(float(h - 1) * 0.82)
    y_e = int(float(h - 1) * 0.91)
    x_s = int(float(w - 1) * 0.41)
    x_e = int(float(w - 1) * 0.51)
    cropped_image = cropped_image[y_s:y_e, x_s:x_e]

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    threshold_value = 55

    _, mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return mask


def get_hook_mask(frame):
    res = cv2.bitwise_and(frame, frame, mask=mask)
    display_frame(res)

    # Try to subtract the background
    bg = cv2.imread('Images/HookBackground.png')
    tolerance = 11
    mask2 = np.all(np.abs(frame - bg) >= tolerance, axis=-1)
    mask_image = (mask2.astype(np.uint8)) * 255
    #display_frame(mask_image)

    # Extract largest contour
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours
    hook_cont = None
    largest_area = 0
    for i, cnt in enumerate(contours):
        # if the contour has no other contours inside of it
        if hierarchy[0][i][2] != -1:
            continue

        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            hook_cont = cnt

    bb = get_bounding_boxes([hook_cont])
    for (x, y, w, h) in bb:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Green color, thickness of 2
        a = int(-6.18 * w + 247.27)
        text = str(a)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)  # White color
        thickness = 2
        position = (x, y)  # Bottom-left corner of the text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    frame = overlay_protractor(frame)

    return mask
