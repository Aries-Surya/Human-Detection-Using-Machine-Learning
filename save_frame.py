import cv2
import os
import time
from ultralytics import YOLO

# Open the RTSP stream
# cap = cv2.VideoCapture("rtsp://192.168.6.129:554/stream0")

cap = cv2.VideoCapture("Input/27-08-2024/Video00001.mp4")

# Directory to save frames
save_dir = "saved_frames"
os.makedirs(save_dir, exist_ok=True)

# Initialize a frame counter
frame_count = 0
save_interval = 1  # Save 1 frame per second

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video stream")
else:
    # Get the FPS of the video stream
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    # Calculate the number of frames to skip to save 1 frame per second
    frame_skip = int(fps * save_interval)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Save every nth frame based on the frame skip
        if frame_count % frame_skip == 0:
            # Save the frame as an image file
            frame_filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
