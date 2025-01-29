from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # You can choose a different variant like 'yolov8s.pt'

model.train(data=r'data-sets\data.yaml', epochs=10, imgsz=640)

metrics = model.val()  # evaluate model performance on the validation set
# results = model("")  # predict on an image
# for result in results:
    # result.show()  # Display the predictions

# path = model.export(format="onnx")  # export the model to ONNX format
