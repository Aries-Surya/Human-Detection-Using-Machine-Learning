from ultralytics import YOLO

# Path to your trained model
model_path = r'D:\MS-Drone Task\HumanDetection\weights\300m.pt'

# Path to your data.yaml file
data_yaml_path = r'D:\MS-Drone Task\HumanDetection\300M-HumanDetection\data.yaml'

# Load the trained model
model = YOLO(model_path)

# Evaluate the model on the validation dataset
results = model.val(data=data_yaml_path)

# Extract and print the evaluation results
print("Model Evaluation Results:")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")
print(f"Precision: {results.box.p:.4f}")
print(f"Recall: {results.box.r:.4f}")
print(f"F1 Score: {results.box.f1:.4f}")
