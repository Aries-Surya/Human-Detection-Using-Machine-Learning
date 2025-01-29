from ultralytics import YOLO
import cv2
import os

# Paths
model_path = r'weights\best(27-08).pt'
input_folder = r'saved_frames'
output_folder = r'Output'
cropped_folder = r'Output\cropped_objects'  # Folder to save cropped objects

# Create output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(cropped_folder, exist_ok=True)

# Load the trained model
model = YOLO(model_path)

# Iterate through images in the input folder
for image_name in os.listdir(input_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        # Perform inference
        results = model.predict(source=image_path)

        # Process results
        annotated_image = False  # Flag to track if any person was detected

        for result in results:
            boxes = result.boxes
            names = result.names

            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bounding box coordinates
                    cls = int(box.cls[0])  # Get class index
                    conf = box.conf[0]  # Get confidence score
                    label = names[cls]

                    if label == 'person' and conf >= 0.3:  # Check if detected object is a person with a confidence score above a threshold
                        annotated_image = True  # Set the flag to True if a person is detected

                        # Draw bounding boxes and labels
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Crop the detected person or object
                        cropped_object = image[y1:y2, x1:x2]

                        # Resize the cropped object to 500x500
                        resized_cropped_object = cv2.resize(cropped_object, (500, 500))

                        # Save the resized cropped object
                        cropped_image_name = f'{os.path.splitext(image_name)[0]}_object_{i+1}.jpg'
                        cropped_image_path = os.path.join(cropped_folder, cropped_image_name)
                        cv2.imwrite(cropped_image_path, resized_cropped_object)
                        print(f'Saved cropped object: {cropped_image_path}')

        # if annotated_image:
        #     # Save the annotated image if any persons were detected
        #     # output_image_path = os.path.join(output_folder, image_name)
        #     # cv2.imwrite(output_image_path, image)
        #     # print(f'Saved annotated image: {output_image_path}')
        # else:
        #     # Optionally delete the image from the input folder if no persons were detected
        #     # os.remove(image_path)
        #     # print(f'No persons detected, deleted image: {image_path}')
