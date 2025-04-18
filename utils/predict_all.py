import os
import glob
import csv
from ultralytics import YOLO
import cv2

# Path configurations
model_path = "models/yolov8n/weights/best.pt"
images_dir = "test/images"
output_dir = "test/predictions"
labels_dir = "test/labels"
csv_output = "test/detection_results.csv"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the YOLOv8n model
model = YOLO(model_path)

# Get all image files
image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
    glob.glob(os.path.join(images_dir, "*.jpeg")) + \
    glob.glob(os.path.join(images_dir, "*.png"))

print(f"Found {len(image_files)} images to process")

# Prepare CSV file
with open(csv_output, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image', 'Detected_Count', 'Ground_Truth_Count'])
    
    # Process each image
    for img_path in image_files:
        # Get the filename without extension
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Run prediction
        results = model(img_path)

        # Get the original image
        img = cv2.imread(img_path)

        # Count detected individuals (with confidence > 0.30)
        detected_count = 0
        
        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Only display predictions with confidence > 0.30 (30%)
                if conf > 0.30:
                    detected_count += 1
                    
                    # Draw rectangle
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Add label with confidence
                    label = f"{conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

        # Get ground truth count
        gt_count = 0
        gt_label_path = os.path.join(labels_dir, f"{name_without_ext}.txt")
        if os.path.exists(gt_label_path):
            with open(gt_label_path, 'r') as f:
                gt_count = len(f.readlines())
        
        # Add count information to the image
        count_text = f"Detected: {detected_count} | Ground Truth: {gt_count}"
        cv2.putText(img, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        # Save the image with predictions
        output_path = os.path.join(output_dir, f"pred_{name_without_ext}.jpg")
        cv2.imwrite(output_path, img)
        
        # Write results to CSV
        csv_writer.writerow([filename, detected_count, gt_count])
        
        # Print minimal progress info
        print(f"Processed {filename}")

print(f"All images processed successfully! Results saved to {csv_output}")
