from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO("models/yolov11s/weights/best.pt")

# Validate the model on your validation dataset
metrics = model.val(
    data="data.yaml", imgsz=640, batch=16, single_cls=True, device="mps")

# Print the mAP75 value
print(f"mAP75: {metrics.box.map75}")
