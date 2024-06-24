import cv2 as cv2
import time
import Helper
import ObjectDetection as OD


def test_image():
    start_time = time.time()
    frame = cv2.imread('Images/screenshot.png')
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


def test_video():
    cap = cv2.VideoCapture("Videos/gameplay.mp4")
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


if __name__ == "__main__":
    #test_image()
    test_video()
