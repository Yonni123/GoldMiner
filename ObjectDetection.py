import cv2 as cv2
import numpy as np
import Helper
from Helper import display_frame


def edge_detection(frame):
    threshold1 = 50
    threshold2 = 2 * threshold1
    edges = cv2.Canny(frame, threshold1, threshold2)

    line_edges = cv2.imread('Images/TheLine.png', cv2.IMREAD_GRAYSCALE)
    line_edges = line_edges[145: Helper.image_height, 0: Helper.image_width]
    edges = cv2.bitwise_and(edges, line_edges)

    kernel = np.ones((10, 10), np.uint8)
    edges2 = cv2.dilate(edges, kernel, iterations=2)
    edges2 = cv2.erode(edges2, kernel, iterations=1)
    line_inv = cv2.bitwise_not(line_edges)
    intersect = cv2.bitwise_and(edges2, line_inv)
    edges = cv2.bitwise_or(edges, intersect)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges


def get_mask(frame):
    height, width = frame.shape[:2]
    cropped = frame[145: height, 0: width]  # Crop the image since we don't want the top part

    edged = edge_detection(cropped)  # Canny edge detection
    # Fix edges of frame
    sides = edged.copy()
    thickness = 5
    sides[thickness:(height-145) - thickness, thickness:width - thickness] = 0
    kernel = np.ones((22, 22), np.uint8)
    sides = cv2.dilate(sides, kernel, iterations=6)
    sides = cv2.erode(sides, kernel, iterations=5)
    thickness = 2
    sides[thickness:(height - 145) - thickness, thickness:width - thickness] = 0
    edged = cv2.add(edged, sides)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)  # Black image
    # loop through the contours
    for i, cnt in enumerate(contours):
        # if the contour has no other contours inside of it
        if hierarchy[0][i][2] == -1 or True:
            cv2.drawContours(mask, [cnt], 0, (255), -1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    uncropped_mask = np.zeros((height, width), dtype=np.uint8)
    uncropped_mask[145: height, 0: width] = mask
    return uncropped_mask


def get_bounding_boxes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
    return boxes