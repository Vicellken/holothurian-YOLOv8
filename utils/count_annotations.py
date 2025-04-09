import os
import glob


def count_annotations(folder_path):
    """
    Count the number of bounding boxes in YOLO txt files
    """
    annotation_files = glob.glob(os.path.join(folder_path, "*.txt"))
    total_files = len(annotation_files)
    total_boxes = 0

    for file_path in annotation_files:
        try:
            # each line in the txt file represents one bounding box
            with open(file_path, 'r') as f:
                lines = f.readlines()
                boxes_in_file = len(lines)
                total_boxes += boxes_in_file
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return total_files, total_boxes


# Folders to process
folders = ["test", "train", "valid"]
base_dir = "PATH_TO_DATASET"

print("Annotation Statistics:")
print("-" * 50)
print(f"{'Folder':<10} {'Files':<10} {'Boxes':<10} {'Avg Boxes/File':<15}")
print("-" * 50)

for folder in folders:
    folder_path = os.path.join(base_dir, folder, "labels")

    # Check if the labels directory exists
    if not os.path.exists(folder_path):
        print(f"{folder:<10} Directory not found: {folder_path}")
        continue

    files, boxes = count_annotations(folder_path)
    avg_boxes = boxes / files if files > 0 else 0

    print(f"{folder:<10} {files:<10} {boxes:<10} {avg_boxes:.2f}")

print("-" * 50)
