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


def get_hook_direction(frame):
    mask = extract_wire_mask(frame)  # Get the mask of the wire that's swinging behind the hook
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours
    except:
        return -1

    # Make sure we actually found something meaningful (otherwise it will crash during loading screen for example)
    if len(contours) == 0:
        return -1
    largest_cont = get_largest_cont(contours)
    if largest_cont is None or len(largest_cont) == 0:
        return -1

    # Fit a line through the white points, and return the unit vector
    [vx, vy, _, _] = cv2.fitLine(largest_cont, cv2.DIST_L2, 0, 0.01, 0.01)
    if vx < 0.05:
        vx = 0
        vy = 1
    return vx, vy, mask


def draw_hook_direction(frame, direction):
    Px, Py = 481, 88  # Coordinates of pixel P
    if type(direction) == int:
        print("Couldn't draw direction, wrong type")
        return frame

    # Calculate parameter t
    if direction[0] != 0:
        t = (frame.shape[0] - Py) / direction[1]  # Use frame height as the 'P_prime' y-coordinate
    else:
        t = 999999999  # Set a large value if direction[1] is zero

    # Calculate endpoint of the line segment
    endpoint_x = int(Px + t * direction[0])
    endpoint_y = int(Py + t * direction[1])

    # Draw a line from P to the calculated endpoint
    cv2.line(frame, (Px, Py), (endpoint_x, endpoint_y), 255, 1)
    return frame
