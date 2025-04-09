from yolo_cam.utils.image import show_cam_on_image
from yolo_cam.eigen_cam import EigenCAM
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import warnings
import os
import glob

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Load models
n_model = YOLO("models/yolov8n/weights/best.pt")
n_model.cpu()
n_target_layers = [n_model.model.model[-4]]

s_model = YOLO("models/yolov8s/weights/best.pt")
s_model.cpu()
s_target_layers = [s_model.model.model[-4]]

m_model = YOLO("models/yolov8m/weights/best.pt")
m_model.cpu()
m_target_layers = [m_model.model.model[-4]]

# Initialize EigenCAM for each model
n_cam = EigenCAM(n_model, n_target_layers, task="od")
s_cam = EigenCAM(s_model, s_target_layers, task="od")
m_cam = EigenCAM(m_model, m_target_layers, task="od")

# Get all bbox images for matching
bbox_images = glob.glob("test/bbox/*.jpg")
bbox_dict = {}
for bbox_path in bbox_images:
    bbox_name = os.path.basename(bbox_path)
    if bbox_name.startswith("gt_"):
        # Extract the base name (e.g., "Stanley_1_18") from complex filenames
        # Format: gt_Stanley_1_18_jpg.rf.a1e42275c35113f4eaefbee788401384.jpg
        # Split after removing "gt_" prefix
        parts = bbox_name[3:].split("_jpg.rf.")
        if len(parts) > 0:
            base_name = parts[0]  # This should be "Stanley_1_18"
            bbox_dict[base_name] = bbox_path

# Function to process a single image


def process_image(img_path, bbox_dict, n_cam, s_cam, m_cam):
    # Get base filename (e.g., "Stanley_1_18") from test image
    img_filename = os.path.basename(img_path)
    # Handle format like: Stanley_1_18_jpg.rf.a1e42275c35113f4eaefbee788401384.jpg
    parts = img_filename.split("_jpg.rf.")
    if len(parts) > 0:
        img_name = parts[0]  # This should be "Stanley_1_18"
    else:
        # Fallback to original method if format is different
        img_name = os.path.basename(img_path).split(".")[0]

    # Load and preprocess image
    img = cv2.imread(img_path)
    rgb_img = img.copy()
    img = np.float32(img) / 255

    # Load ground truth image if available
    if img_name in bbox_dict:
        gt = cv2.imread(bbox_dict[img_name])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: No bbox found for {img_name}")
        # Create a blank image with same dimensions as the original
        gt = np.zeros_like(img)

    # Process with YOLOv8n
    n_grayscale_cam = n_cam(rgb_img)[0, :, :]
    n_cam_image = show_cam_on_image(img, n_grayscale_cam, use_rgb=True)

    # Process with YOLOv8s
    s_grayscale_cam = s_cam(rgb_img)[0, :, :]
    s_cam_image = show_cam_on_image(img, s_grayscale_cam, use_rgb=True)

    # Process with YOLOv8m
    m_grayscale_cam = m_cam(rgb_img)[0, :, :]
    m_cam_image = show_cam_on_image(img, m_grayscale_cam, use_rgb=True)

    # Create a 2x2 grid for the images
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Add titles and images to the subplots
    axs[0, 0].set_title("Ground Truth")
    axs[0, 0].imshow(gt)
    axs[0, 0].axis("off")

    axs[0, 1].set_title("YOLOv8n")
    axs[0, 1].imshow(n_cam_image)
    axs[0, 1].axis("off")

    axs[1, 0].set_title("YOLOv8s")
    axs[1, 0].imshow(s_cam_image)
    axs[1, 0].axis("off")

    axs[1, 1].set_title("YOLOv8m")
    axs[1, 1].imshow(m_cam_image)
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"plot_{img_name}.png", dpi=600, transparent=True, bbox_inches="tight")
    plt.close()

    print(f"Processed {img_name}")
    return img_name


if __name__ == "__main__":
    # Get all images from test directory
    test_images = glob.glob("test/images/*.jpg")

    print(f"Processing {len(test_images)} images sequentially")

    results = []
    for img_path in test_images:
        try:
            result = process_image(img_path, bbox_dict, n_cam, s_cam, m_cam)
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    print(f"Completed processing {len(results)} images")
