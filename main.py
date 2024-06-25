import cv2 as cv2
import time
import Helper
import ObjectDetection as OD
import mss
import numpy as np


def test_image(filename):
    start_time = time.time()
    frame = cv2.imread(filename)
    frame = Helper.preprocess_frame(frame)

    mask = OD.get_mask(frame)
    boxes = OD.get_bounding_boxes(mask)
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness of 2
    end_time = time.time()
    print(f"Took {end_time - start_time:.6f} seconds to run.")

    cv2.imshow('res', frame)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_video(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_counter = 0  # Initialize frame counter
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Reached the end of the video.")
            break

        frame = Helper.preprocess_frame(frame)
        mask = OD.get_mask(frame)
        boxes = OD.get_bounding_boxes(mask)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness of 2

        # Display the resulting frame
        cv2.imshow('Video Playback', frame)
        cv2.imshow('Mask Playback', mask)
        frame_counter += 1

        # Exit the video when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def test_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while True:
            frame = sct.grab(monitor)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = frame[186:845,160:1155]

            frame = Helper.preprocess_frame(frame)

            mask = OD.get_mask(frame)
            boxes = OD.get_bounding_boxes(mask)
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness of 2

            cv2.imshow('Screen Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #test_image('Images/lvl3manyobjects.png')
    test_video('Videos/lvl7.mp4')
    #test_screen()
