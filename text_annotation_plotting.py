import json
import os
import cv2
import random
# from PIL import ExifTags, Image
import numpy as np
from tqdm import tqdm 

print("Start")

#### ------------------------------------------------------- dir ------------------------------
for subfolderx in ["train","test"]:

    # Load the annotations from the provided text file
    annotations_file_path = f"/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text/{subfolderx}.txt"
    print(f"Loading annotations from: {annotations_file_path}")

    # Directory where the images are located
    img_dir = f"/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text/{subfolderx}"
    print(f"Images directory: {img_dir}")

    # Process each image file
    output_dir = f"/home/frinksserver/subhra/paddleOCR_data_preparation/detection_data_check_plotted/{subfolderx}"
    os.makedirs(output_dir, exist_ok=True)

    ### ------------------------------------------------------------------------------------
    with open(annotations_file_path, 'r') as file:
        annotations = file.readlines()

    # Print first few lines of the annotation file for debugging
    print("First few lines of the annotation file:")
    for line in annotations[:5]:
        print(line.strip())

    annotations_dict = {}
    for annotation in annotations:
        if "\t" in annotation:
            img_path, details = annotation.split('\t')
            img_name = os.path.basename(img_path.strip())
            if img_name in annotations_dict:
                annotations_dict[img_name].extend(json.loads(details.strip()))
            else:
                annotations_dict[img_name] = json.loads(details.strip())

    # Print a few entries from the annotations dictionary for debugging
    print("Sample entries from annotations dictionary:")
    for k, v in list(annotations_dict.items())[:3]:
        print(f"{k}: {v}")


    # List all image files in the directory
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

    # Print a few sample image file names for debugging
    print("Sample image file names from the directory:")
    for img_name in random.sample(image_files, min(3, len(image_files))):
        print(img_name)


    output_files = []
    for img_name in tqdm(image_files):
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            image = cv2.imread(img_path)


            # Get image dimensions
            img_height, img_width = image.shape[:2]

            # Get the annotation for the specific image
            annotation = annotations_dict.get(img_name, [])

            # Draw the bounding boxes and labels
            for item in annotation:
                points = item['points']
                transcription = item['transcription']

                # Draw the bounding box
                points_np = np.array(points, np.int32)
                points_np = points_np.reshape((-1, 1, 2))
                cv2.polylines(image, [points_np], isClosed=True, color=(0, 0, 255), thickness=3)

                x0, y0 = points[0]
                font_scale = min(img_width, img_height) / 500  # Increased font scale
                font_thickness = max(2, int(min(img_width, img_height) / 1000))  # Increased font thickness
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(transcription, font, font_scale, font_thickness)
                text_w, text_h = text_size
                cv2.rectangle(image, (x0, y0 - text_h - 10), (x0 + text_w, y0), (0, 0, 255), -1)
                cv2.putText(image, transcription, (x0, y0 - 5), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

            # Save the modified image
            output_path = os.path.join(output_dir, f"annotated_{img_name}")
            cv2.imwrite(output_path, image)
            output_files.append(output_path)

    print("Output files:")
    print(output_files)
