import json
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def convert_paddle_to_yolo(points, img_width, img_height):
    x_coords, y_coords = zip(*points)
    x_center = (min(x_coords) + max(x_coords)) / (2 * img_width)
    y_center = (min(y_coords) + max(y_coords)) / (2 * img_height)
    width = (max(x_coords) - min(x_coords)) / img_width
    height = (max(y_coords) - min(y_coords)) / img_height
    return [0, x_center, y_center, width, height]  # 0 is the class id for 'text'

def process_paddle_file(input_file, yolo_output_dir):
    print(f"Processing file: {input_file}")
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text", "supercategory": "text"}]
    }
    
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    annotation_id = 1
    for image_id, line in tqdm(enumerate(lines, start=1), total=len(lines), desc="Processing images"):
        image_path, annotations_json = line.strip().split('\t')
        annotations = json.loads(annotations_json)
        
        # Get actual image dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping...")
            continue
        img_height, img_width = img.shape[:2]
        
        # COCO image info
        file_name = os.path.basename(image_path)
        coco_format["images"].append({
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": file_name
        })
        
        # YOLO format
        yolo_annotations = []
        
        for annotation in annotations:
            points = annotation["points"]
            
            # YOLO conversion
            yolo_bbox = convert_paddle_to_yolo(points, img_width, img_height)
            yolo_annotations.append(" ".join(map(str, yolo_bbox)))
            
            # COCO conversion
            x_coords, y_coords = zip(*points)
            x_min, y_min = min(x_coords), min(y_coords)
            width, height = max(x_coords) - x_min, max(y_coords) - y_min
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "segmentation": [[coord for point in points for coord in point]],
                "iscrowd": 0
            })
            annotation_id += 1
        
        # Save YOLO annotation
        yolo_file_name = os.path.splitext(file_name)[0] + '.txt'
        yolo_file_path = os.path.join(yolo_output_dir, yolo_file_name)
        with open(yolo_file_path, 'w') as yolo_file:
            yolo_file.write("\n".join(yolo_annotations))
    
    print(f"Processed {len(coco_format['images'])} images and {len(coco_format['annotations'])} annotations")
    return coco_format

def draw_coco_annotations(image_path, annotations, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping visualization...")
        return
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

def draw_yolo_annotations(image_path, annotation_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping visualization...")
        return
    height, width = img.shape[:2]
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        x = int((x_center - w/2) * width)
        y = int((y_center - h/2) * height)
        w = int(w * width)
        h = int(h * height)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

def create_annotations(input_dir, output_dir):
    print("Starting annotation creation process...")
    output_dir.mkdir(exist_ok=True)

    for split in ['train', 'test']:
        print(f"Processing {split} split...")
        # Create output directories
        (output_dir / split).mkdir(exist_ok=True)
        
        # Process files
        input_file = input_dir / f"{split}.txt"
        yolo_output_dir = output_dir / split
        coco_data = process_paddle_file(input_file, yolo_output_dir)
        
        # Save Faster R-CNN JSON
        faster_rcnn_output_file = output_dir / f"{split}_fasterrcnn.json"
        with open(faster_rcnn_output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"Saved Faster R-CNN annotations to {faster_rcnn_output_file}")

    print("Annotation files created successfully.")

def visualize_annotations(input_dir, output_dir, faster_rcnn_check_dir, yolo_check_dir):
    print("Starting annotation visualization process...")
    faster_rcnn_check_dir.mkdir(exist_ok=True)
    yolo_check_dir.mkdir(exist_ok=True)

    for split in ['train', 'test']:
        print(f"Visualizing {split} split annotations...")
        (faster_rcnn_check_dir / split).mkdir(exist_ok=True)
        (yolo_check_dir / split).mkdir(exist_ok=True)

        # Load Faster R-CNN JSON
        faster_rcnn_file = output_dir / f"{split}_fasterrcnn.json"
        with open(faster_rcnn_file, 'r') as f:
            coco_data = json.load(f)

        for image_info in tqdm(coco_data['images'], desc=f"Visualizing {split} annotations"):
            image_id = image_info['id']
            file_name = image_info['file_name']
            image_path = input_dir / split / file_name

            # Faster R-CNN annotations
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
            faster_rcnn_output_path = faster_rcnn_check_dir / split / f"faster_rcnn_{file_name}"
            draw_coco_annotations(str(image_path), annotations, str(faster_rcnn_output_path))

            # YOLO annotations
            yolo_annotation_path = output_dir / split / f"{os.path.splitext(file_name)[0]}.txt"
            yolo_output_path = yolo_check_dir / split / f"yolo_{file_name}"
            draw_yolo_annotations(str(image_path), str(yolo_annotation_path), str(yolo_output_path))

    print("Annotation visualization completed. Check the output folders for results.")

def main():
    """
    - creates annotations for yolo
    - creates annotations for fasterrcnn
    - plot annotations (yolo and fasterrcnn) on images to check quality
    """
    # Define directories
    input_dir = Path("split_det_text")
    output_dir = Path("yolo_split_det_text")
    faster_rcnn_check_dir = Path("faster_rcnn_annotation_check")
    yolo_check_dir = Path("yolo_annotation_check")

    # Create annotations
    create_annotations(input_dir, output_dir)

    # Visualize annotations
    visualize_annotations(input_dir, output_dir, faster_rcnn_check_dir, yolo_check_dir)

if __name__ == "__main__":
    main()