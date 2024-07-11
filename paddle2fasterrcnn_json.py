'''
This code converts paddle format detection to fastercnn json
input: 
    train_file_path = "/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text/train.txt"
out:
    output_file_path = "/home/frinksserver/subhra/paddleOCR_data_preparation/instances_default_converted.json"
'''

import json
import os

# Function to convert train.txt format to COCO format
def convert_to_coco(train_file_path, output_file_path):
    # Initialize COCO format structure
    coco_format = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "Frinks AI",
            "date_created": "2024-06-04",
            "description": "OCR dataset",
            "url": "",
            "version": "1.0",
            "year": 2024
        },
        "categories": [{"id": 1, "name": "text", "supercategory": "text"}],
        "images": [],
        "annotations": []
    }
    
    # Read train.txt file
    with open(train_file_path, 'r') as file:
        lines = file.readlines()
    
    annotation_id = 1
    for line in lines:
        # Split line into image path and annotations
        image_path, annotations_json = line.strip().split('\t')
        annotations = json.loads(annotations_json)
        
        # Extract image info
        image_id = len(coco_format["images"]) + 1
        file_name = os.path.basename(image_path)
        image_info = {
            "id": image_id,
            "width": 0,  # Placeholder, replace with actual width
            "height": 0,  # Placeholder, replace with actual height
            "file_name": file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        coco_format["images"].append(image_info)
        
        for annotation in annotations:
            # Convert points to segmentation format and calculate bbox
            points = annotation["points"]
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min
            
            bbox = [x_min, y_min, width, height]
            segmentation = [coord for point in points for coord in point]
            area = width * height
            
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {
                    "occluded": False,  # Placeholder, replace with actual attribute
                    "rotation": 0.0  # Placeholder, replace with actual attribute
                }
            }
            coco_format["annotations"].append(annotation_info)
            annotation_id += 1
    
    # Write to output JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(coco_format, output_file, indent=4)

# Paths to input and output files
train_file_path = "/home/frinksserver/backup_server3/fasterrcnn_training/skh_ocr_data/combined_paddle.txt"
output_file_path = "/home/frinksserver/backup_server3/fasterrcnn_training/skh_ocr_data/combined.json"

# Convert the format
convert_to_coco(train_file_path, output_file_path)
