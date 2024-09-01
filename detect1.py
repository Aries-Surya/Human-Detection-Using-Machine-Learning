from ultralytics import YOLO
import os, cv2

# Load the trained YOLOv8 model
#model = YOLO("/home/casr-3/Downloads/human_dataset.v1i.yolov8/human_detect3/weights/best.pt")##downloaded_dataset
#model = YOLO("/home/casr-3/Downloads/human_dataset.v1i.yolov8/human_detect5/weights/best.pt")####no detection with only statue
#model = YOLO("/home/casr-3/Documents/300m_3human_dataset/300m_3human_dataset8/weights/last.pt")
# model = YOLO("/home/casr-3/Documents/50m_3human_dataset/50m_3human_dataset2/weights/best.pt")
model = YOLO(r"D:\MS-Drone Task\HumanDetection\weights\best.pt")

# Set paths
image_folder_path = r"D:\MS-Drone Task\HumanDetection\300M-HumanDetection\train\images"
output_folder_path = r"D:\MS-Drone Task\HumanDetection\output"

# Ensure output folder exists
os.makedirs(output_folder_path, exist_ok=True)

def process_image(image_path):
    try:
        # Perform inference
        results = model(image_path)
        
        for i, result in enumerate(results):
            # Check if any objects were detected
            if result.boxes:  # If there are any detected boxes, save the image
                # Generate output file path
                output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_result_{i}.jpg")
                print("output_file_path", output_file_path)
                
                # Save the annotated image
                annotated_image = result.plot()  # Plot returns the image with detections
                
                # Save the annotated image manually
                success = cv2.imwrite(output_file_path, annotated_image)
                if not success:
                    print(f"Failed to save image: {output_file_path}")
            else:
                print(f"No detection in image: {image_path}, skipping save.")
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


# Process each image in the folder
for image_name in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_name)
    if os.path.isfile(image_path):
        process_image(image_path)

print("Detection completed and results saved.")

