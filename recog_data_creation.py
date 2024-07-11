import os
import json
import cv2
from tqdm import tqdm

# Paths
input_master_folder = '/home/frinksserver/Deepak/OCR/data_preparation/paddleOCR_data_preparation/split_det_text'
output_master_folder = '/home/frinksserver/Deepak/OCR/data_preparation/paddleOCR_data_preparation/split_recog_text'

# Ensure output directories exist
os.makedirs(output_master_folder, exist_ok=True)

# Get the list of subdirectories from input_master_folder
subdirectories = [d for d in os.listdir(input_master_folder) if os.path.isdir(os.path.join(input_master_folder, d))]

def process_images(input_folder, output_folder, annotations_file, output_txt_file):
    # Ensure output directories exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the annotations file
    with open(annotations_file, 'r') as file:
        lines = file.readlines()
    
    # Prepare the output text file
    with open(output_txt_file, 'w') as txt_file:
        # Process each line in the annotations file
        for line in tqdm(lines, desc=f"Processing images in {os.path.basename(input_folder)}"):
            parts = line.split('\t')
            image_path = parts[0]
            annotations = json.loads(parts[1])
            
            # Full image path
            full_image_path = os.path.join(input_folder, os.path.basename(image_path))
            
            # Verify image exists
            if not os.path.exists(full_image_path):
                print(f"Image {full_image_path} not found.")
                continue
            
            # Read the image
            image = cv2.imread(full_image_path)
            
            # Process each annotation
            for idx, annotation in enumerate(tqdm(annotations, desc=f"Processing annotations for {os.path.basename(image_path)}", leave=False)):
                transcription = annotation['transcription']
                points = annotation['points']
                
                # Convert points to integers
                points = [(int(x), int(y)) for x, y in points]
                
                # Get the bounding box coordinates
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Crop the image
                cropped_image = image[y_min:y_max, x_min:x_max]
                
                # Prepare the cropped image filename
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                cropped_image_name = f"{base_filename}_{idx + 1}.png"
                cropped_image_path = os.path.join(output_folder, cropped_image_name)
                
                # Save the cropped image
                cv2.imwrite(cropped_image_path, cropped_image)
                
                # Write the annotation to the output text file
                txt_file.write(f"{cropped_image_path}\t{transcription}\n")

# Process each subdirectory
for subdirectory in subdirectories:
    input_folder = os.path.join(input_master_folder, subdirectory)
    output_folder = os.path.join(output_master_folder, subdirectory)
    annotations_file = os.path.join(input_master_folder, f"{subdirectory}.txt")
    output_txt_file = os.path.join(output_master_folder, f"{subdirectory}.txt")
    
    if os.path.exists(input_folder) and os.path.exists(annotations_file):
        process_images(input_folder, output_folder, annotations_file, output_txt_file)
    else:
        print(f"Directory {input_folder} or file {annotations_file} does not exist.")

print("All images have been processed successfully.")
