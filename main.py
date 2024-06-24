import cv2 as cv2
import numpy as np
import time
import random
import string
from matplotlib import pyplot as plt

black_border_cont = None
image_height = 720
image_width = None


def display_frame(frame):
    # Generate a random window name
    length = 8
    letters = string.ascii_letters
    window_name = ''.join(random.choice(letters) for i in range(length))

    # Display the frame in a window
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_detection(frame):
    threshold1 = 50
    threshold2 = 2 * threshold1
    edges = cv2.Canny(frame, threshold1, threshold2)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges


def calc_black_border_cont(image):
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
    global black_border_cont, image_height, image_width

    if frame is None:
        raise ValueError("Could not open or find the image.")

    # Resize image and keep aspect ratio
    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = int(image_height * aspect_ratio)
    frame = cv2.resize(frame, (new_width, image_height), interpolation=cv2.INTER_AREA)
    image_width = new_width

    # Remove black borders
    if black_border_cont is None:
        calc_black_border_cont(frame)

    if black_border_cont != -1:
        (x, y, w, h) = black_border_cont
        clip_sides = 6
        frame = frame[0: y + h, x + clip_sides: x + w - clip_sides]


    return frame


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


start_time = time.time()
frame = cv2.imread('lvl3.png')
frame = preprocess_frame(frame)

mask = get_mask(frame)
boxes = get_bounding_boxes(mask)
for (x, y, w, h) in boxes:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green color, thickness of 2
end_time = time.time()
print(f"Took {end_time - start_time:.6f} seconds to run.")

cv2.imshow('res', frame)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

cap = cv2.VideoCapture("gameplay.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_counter = 0  # Initialize frame counter
boxes = []
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Delay between frames in milliseconds
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Reached the end of the video.")
        break

    start_time = time.time()
    frame = preprocess_frame(frame)
    mask = get_mask(frame)
    boxes = get_bounding_boxes(mask)
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness of 2

    # Display the resulting frame
    cv2.imshow('Video Playback', frame)
    cv2.imshow('Mask Playback', mask)
    frame_counter += 1
    end_time = time.time()

    # Exit the video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
