import cv2
import os

def video_to_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = video_fps  # Save 1 frame per second

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save one frame per second
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame5_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            print(f"Saved: {frame_filename}")

        frame_count += 1

    cap.release()
    print("Finished converting video to frames.")

# Example usage
video_path = r'D:\MS-Drone Task\HumanDetection\Gimbal_video_dataset\26-08-2024\Video00007.mp4'
output_folder = r'D:\MS-Drone Task\HumanDetection\Gimbal_video_dataset\Raw Img'

video_to_frames(video_path, output_folder)
