from ultralytics import YOLO
import cv2
import os

# Paths
model_path = r'weights\best(27-08).pt'
input_video_path = r'Input\Video00002.mp4'
output_video_path = r'Output\output_video.mp4'

# Load the trained model
model = YOLO(model_path)

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {input_video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the output resolution
output_width, output_height = 1080, 720

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Resize the frame to 1080x720p
    frame = cv2.resize(frame, (output_width, output_height))

    # Perform inference
    results = model.predict(source=frame)

    # Process results
    annotated_frame = False  # Flag to track if any object was detected

    for result in results:
        boxes = result.boxes
        names = result.names

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box coordinates
                cls = int(box.cls[0])  # Get class index
                conf = box.conf[0]  # Get confidence score
                label = names[cls]

                # Check if detected object is one of the specified classes with a confidence score above a threshold
                if conf >= 0.3:
                    annotated_frame = True  # Set the flag to True if an object is detected
                    
                    # Draw bounding boxes and labels
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow('Real-time Detection', frame)

    # Write the frame to the output video
    out.write(frame)

    # Check for 'q' key to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f'Output video saved to {output_video_path}')
