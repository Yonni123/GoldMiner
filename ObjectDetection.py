import cv2 as cv2
import numpy as np
import Helper
from Helper import display_frame


def edge_detection(frame):
    threshold1 = 50
    threshold2 = 2 * threshold1
    edges = cv2.Canny(frame, threshold1, threshold2)

    # Remove background lines
    line_edges = cv2.imread('Images/TheLine.png', cv2.IMREAD_GRAYSCALE)
    line_edges = line_edges[145: Helper.image_height, 0: Helper.image_width]
    edges = cv2.bitwise_and(edges, line_edges)

    # Erode and Dilate the image to make detections better, and to intersect the line_edges by making them bigger
    kernel = np.ones((10, 10), np.uint8)
    edges2 = cv2.dilate(edges, kernel, iterations=2)
    edges2 = cv2.erode(edges2, kernel, iterations=1)

    # Connect the objects that intersected the line_edges after the dilation, to detect them as single object
    line_inv = cv2.bitwise_not(line_edges)
    intersect = cv2.bitwise_and(edges2, line_inv)
    edges = cv2.bitwise_or(edges, intersect)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges


def get_mask(frame):
    # We need to remove "y_shift" pixels from top of the image, since objects aren't there
    y_shift = 145
    height, width = frame.shape[:2]
    cropped = frame[y_shift: height, 0: width]  # Crop the image since we don't want the top part

    edged = edge_detection(cropped)  # Use Canny edge detection and remove background lines

    # Fix edges of frame, objects that are partially outside the frame needs to be closed off so they form object
    sides = edged.copy()
    thickness = 5
    sides[thickness:(height-y_shift) - thickness, thickness:width - thickness] = 0
    kernel = np.ones((22, 22), np.uint8)
    sides = cv2.dilate(sides, kernel, iterations=6)
    sides = cv2.erode(sides, kernel, iterations=5)
    thickness = 2
    sides[thickness:(height - y_shift) - thickness, thickness:width - thickness] = 0
    edged = cv2.add(edged, sides)

    # All objects are detected in a closed loop, so detect these contours and fill them with white to create a mask
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    conts_to_return = []
    for i, cnt in enumerate(contours):
        # if the contour has no other contours inside of it, and is big enough, it's an object
        area = cv2.contourArea(cnt)
        if hierarchy[0][i][2] == -1 and area > 200:
            cv2.drawContours(mask, [cnt], 0, (255), -1)
            for point in cnt:
                point[0][1] += y_shift  # These detected contours are shifted since we cropped the frame, correct for it
            conts_to_return.append(cnt)

    # When object detection is done, we need to shift the image back to its original pos, "uncrop" the frame
    uncropped_mask = np.zeros((height, width), dtype=np.uint8)
    uncropped_mask[y_shift: height, 0: width] = mask
    return uncropped_mask, conts_to_return


def get_bounding_boxes(cont):
    boxes = []
    for contour in cont:
        area = cv2.contourArea(contour)
        if area > 200:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
    return boxes