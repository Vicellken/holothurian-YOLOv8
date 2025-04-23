import os
import glob
import csv
import time
from ultralytics import YOLO
import cv2
import numpy as np

# Path configurations
model_path = "models/yolov8n/weights/best.pt"

# Function to process a dataset directory


def process_dataset(dataset_name):
    # Set up paths for this dataset
    images_dir = f"{dataset_name}/images"
    output_dir = f"{dataset_name}/predictions"
    labels_dir = f"{dataset_name}/labels"
    csv_output = f"{dataset_name}/detection_results.csv"

    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(
            f"Warning: {images_dir} directory does not exist. Skipping {dataset_name} dataset.")
        return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
        glob.glob(os.path.join(images_dir, "*.jpeg")) + \
        glob.glob(os.path.join(images_dir, "*.png"))

    if not image_files:
        print(
            f"Warning: No images found in {images_dir}. Skipping {dataset_name} dataset.")
        return None

    print(f"\nProcessing {dataset_name} dataset:")
    print(f"Found {len(image_files)} images to process")

    # Initialize timing variables
    dataset_start_time = time.time()
    prediction_times = []

    # Prepare CSV file
    with open(csv_output, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['Image', 'Detected_Count', 'Ground_Truth_Count', 'Prediction_Time_ms'])

        # Process each image
        for img_path in image_files:
            # Get the filename without extension
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Start timing for prediction
            pred_start_time = time.time()

            # Run prediction
            results = model(img_path)

            # End timing for prediction
            pred_end_time = time.time()
            pred_time_ms = (pred_end_time - pred_start_time) * \
                1000  # Convert to milliseconds
            prediction_times.append(pred_time_ms)

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
            output_path = os.path.join(
                output_dir, f"pred_{name_without_ext}.jpg")
            cv2.imwrite(output_path, img)

            # Write results to CSV
            csv_writer.writerow(
                [filename, detected_count, gt_count, f"{pred_time_ms:.2f}"])

            # Print minimal progress info
            print(f"Processed {filename} in {pred_time_ms:.2f} ms")

    # Calculate total processing time
    dataset_end_time = time.time()
    dataset_time = dataset_end_time - dataset_start_time

    # Calculate timing statistics
    if prediction_times:
        avg_pred_time = np.mean(prediction_times)
        std_pred_time = np.std(prediction_times)
        min_pred_time = np.min(prediction_times)
        max_pred_time = np.max(prediction_times)

        # Print timing information
        print(f"\n{dataset_name} Timing Statistics:")
        print(f"Total processing time: {dataset_time:.2f} seconds")
        print(f"Average prediction time: {avg_pred_time:.2f} ms")
        print(f"Standard deviation: {std_pred_time:.2f} ms")
        print(f"Minimum prediction time: {min_pred_time:.2f} ms")
        print(f"Maximum prediction time: {max_pred_time:.2f} ms")

        print(
            f"\n{dataset_name} dataset processed successfully! Results saved to {csv_output}")
        return dataset_time
    else:
        print(f"No predictions were made for {dataset_name} dataset.")
        return None


# Main execution
if __name__ == "__main__":
    # Load the YOLOv8n model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Initialize timing variables
    total_start_time = time.time()

    # Process both valid and test datasets
    valid_time = process_dataset("valid")
    test_time = process_dataset("test")

    # Calculate total processing time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Print overall summary
    print("\n===== Overall Summary =====")
    print(f"Total execution time: {total_time:.2f} seconds")

    if valid_time:
        print(f"Valid dataset processing time: {valid_time:.2f} seconds")
    else:
        print("Valid dataset was not processed.")

    if test_time:
        print(f"Test dataset processing time: {test_time:.2f} seconds")
    else:
        print("Test dataset was not processed.")

    print("\nPrediction completed for all available datasets!")
