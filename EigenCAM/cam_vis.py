
import matplotlib.pyplot as plt
import cv2
import os
import glob
from yolo_cam.utils.image import show_cam_on_image
from yolo_cam.eigen_cam import EigenCAM
import numpy as np
from ultralytics import YOLO
import warnings
import multiprocessing
from functools import partial

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Paths
images_dir = "test/images"
bbox_dir = "test/bbox"
predictions_dir = "test/predictions"
output_dir = "comparison_plots"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8n model for EigenCAM
n_model = YOLO("models/yolov8n/weights/best.pt")
n_target_layers = [n_model.model.model[-4]]
n_cam = EigenCAM(n_model, n_target_layers, task="od")

# Get all bbox images for matching
bbox_images = glob.glob(os.path.join(bbox_dir, "*.jpg"))
bbox_dict = {}
for bbox_path in bbox_images:
    bbox_name = os.path.basename(bbox_path)
    if bbox_name.startswith("gt_"):
        # Extract the base name from complex filenames
        parts = bbox_name[3:].split("_jpg.rf.")
        if len(parts) > 0:
            base_name = parts[0]  # This should be "Stanley_1_18"
            bbox_dict[base_name] = bbox_path

# Get all prediction images for matching
pred_images = glob.glob(os.path.join(predictions_dir, "*.jpg"))
pred_dict = {}
for pred_path in pred_images:
    pred_name = os.path.basename(pred_path)
    if pred_name.startswith("pred_"):
        # Extract the base name from complex filenames
        parts = pred_name[5:].split("_jpg.rf.")  # Remove "pred_" prefix
        if len(parts) > 0:
            base_name = parts[0]  # This should be "Stanley_1_18"
            pred_dict[base_name] = pred_path


def process_image(img_path, bbox_dict, pred_dict, n_cam):
    # Get base filename from test image
    img_filename = os.path.basename(img_path)
    parts = img_filename.split("_jpg.rf.")
    if len(parts) > 0:
        img_name = parts[0]  # This should be "Stanley_1_18"
    else:
        img_name = os.path.basename(img_path).split(".")[0]

    try:
        # Load and preprocess image
        img = cv2.imread(img_path)
        rgb_img = img.copy()
        img_float = np.float32(img) / 255

        # 1. Ground Truth - Load from bbox dictionary
        gt_path = bbox_dict.get(img_name)
        if gt_path:
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        else:
            print(f"Warning: No bbox found for {img_name}")
            gt = np.zeros_like(img)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        # 2. EigenCAM visualization
        n_grayscale_cam = n_cam(rgb_img)[0, :, :]
        n_cam_image = show_cam_on_image(
            img_float, n_grayscale_cam, use_rgb=True)

        # 3. Prediction - Load from predictions dictionary
        pred_path = pred_dict.get(img_name)
        if pred_path and os.path.exists(pred_path):
            predict = cv2.imread(pred_path)
            predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        else:
            print(f"Warning: No prediction found for {img_name}")
            predict = np.zeros_like(img)
            predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)

        # Create a 1x3 grid for the images
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

        ax1, ax2, ax3 = axs

        # Add titles and images to the subplots
        ax1.set_title('Ground Truth')
        ax1.imshow(gt)
        ax1.axis("off")

        ax2.set_title('EigenCAM (YOLOv8n)')
        ax2.imshow(n_cam_image)
        ax2.axis("off")

        ax3.set_title('Prediction')
        ax3.imshow(predict)
        ax3.axis("off")

        plt.savefig(
            os.path.join(output_dir, f"comparison_{img_name}.png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close()

        print(f"Saved comparison for {img_name}")
        return img_name
    except Exception as e:
        print(f"Error processing {img_name}: {str(e)}")
        return None


if __name__ == "__main__":
    # Get all images from test directory
    test_images = glob.glob(os.path.join(images_dir, "*.jpg"))

    # Create a partial function with fixed arguments
    process_func = partial(
        process_image,
        bbox_dict=bbox_dict,
        pred_dict=pred_dict,
        n_cam=n_cam
    )

    # Determine number of processes (use number of CPU cores)
    num_processes = max(1, multiprocessing.cpu_count() -
                        1)  # Leave one core free
    print(
        f"Processing {len(test_images)} images using {num_processes} processes")

    # Create a pool of workers and map the processing function to the images
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, test_images)

    # Filter out None results (failed processing)
    results = [r for r in results if r is not None]
    print(f"Completed processing {len(results)} images")
