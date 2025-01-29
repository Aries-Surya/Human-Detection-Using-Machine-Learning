from ultralytics import YOLO
import cv2
import os

# Paths
model_path = r'weights\best2.pt'
input_video_path = r'Input\27-08-2024\Video00001.mp4'
output_video_path = r'Output\output_video.mp4'
cropped_images_folder = r'Output\cropped_images'

# Create the folder to save cropped images if it doesn't exist
os.makedirs(cropped_images_folder, exist_ok=True)

# Load the trained model
model = YOLO(model_path)

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {input_video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0  # Initialize frame counter

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform inference
    results = model.predict(source=frame)

    # Process results
    for result in results:
        boxes = result.boxes
        names = result.names

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bounding box coordinates
                cls = int(box.cls[0])  # Get class index
                conf = box.conf[0]  # Get confidence score
                label = names[cls]

                # Check if detected object is one of the specified classes with a confidence score above a threshold
                if conf >= 0.3:
                    # Draw bounding boxes and labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Crop the detected object
                    cropped_image = frame[y1:y2, x1:x2]

                    # Check if the cropped image is valid before resizing
                    if cropped_image is not None and cropped_image.size != 0:
                        try:
                            # Resize the cropped image to 50x50
                            resized_cropped_image = cv2.resize(cropped_image, (50, 50), interpolation=cv2.INTER_AREA)

                            # Save the resized image
                            cropped_image_filename = os.path.join(cropped_images_folder, f'{label}_frame{frame_count}_object{i}.jpg')
                            cv2.imwrite(cropped_image_filename, resized_cropped_image)
                        except cv2.error as e:
                            print(f"Error resizing image: {e}")
                    else:
                        print(f"Invalid crop at frame {frame_count}, object {i}. Skipping.")

    # Display the frame in a window
    cv2.imshow('Real-time Detection', frame)

    # Write the frame to the output video
    out.write(frame)

    # Increment frame count
    frame_count += 1

    # Check for 'q' key to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f'Output video saved to {output_video_path}')
print(f'Cropped images saved to {cropped_images_folder}')
