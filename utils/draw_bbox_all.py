import cv2
import os


def draw_boxes(image_path, annotation_path, output_path):
    # Read the image
    img = cv2.imread(image_path)

    # Open the YOLO-formatted annotation file
    with open(annotation_path, 'r') as file:
        for line in file:
            # Parse the YOLO annotation
            class_label, x_center, y_center, width, height = map(
                float, line.split())

            # Calculate bounding box coordinates
            x, y, w, h = int((x_center - width / 2) * img.shape[1]), \
                int((y_center - height / 2) * img.shape[0]), \
                int(width * img.shape[1]), \
                int(height * img.shape[0])

            # Draw the bounding box on the image
            color = (160, 32, 240)  # Purple
            thickness = 4
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

            # Add class label; NA in this case as we only have one class
            label = f"{int(class_label)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0  # remove label
            font_thickness = 0  # remove label
            text_size = cv2.getTextSize(
                label, font, font_scale, font_thickness)[0]
            img = cv2.putText(img, label, (x, y - 5), font,
                              font_scale, color, font_thickness)

    cv2.imwrite(output_path, img)


# EDIT THESE TO YOUR OWN PATHS
txt_folder = "test/labels"
img_folder = "test/images"
output_folder = "test/bbox"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all txt files and corresponding image files
txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
img_files = [f for f in os.listdir(
    img_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

# Iterate through each pair
for txt_file in txt_files:
    # Assuming corresponding image file has the same name with different extension
    img_file = os.path.splitext(txt_file)[0] + ".jpg"

    txt_path = os.path.join(txt_folder, txt_file)
    img_path = os.path.join(img_folder, img_file)
    output_name = "gt_" + img_file
    output_path = os.path.join(output_folder, output_name)

    draw_boxes(img_path, txt_path, output_path)
