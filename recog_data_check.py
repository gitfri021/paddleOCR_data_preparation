import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
input_master_folder = '/home/frinksserver/subhra/paddleOCR_data_preparation/split_recog_text'  # Update with the actual path
output_master_folder = '/home/frinksserver/subhra/paddleOCR_data_preparation/recog_data_check'

# Ensure output directory exists
os.makedirs(output_master_folder, exist_ok=True)

def create_canvas(image_path, transcription):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Set font parameters
    font_scale = 1
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size
    text_size, _ = cv2.getTextSize(transcription, font, font_scale, font_thickness)
    text_width, text_height = text_size

    # Create a white canvas
    canvas_height = max(image_height, text_height + 20)
    canvas_width = image_width + text_width + 40  # Add extra width for text
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Place the image onto the canvas
    canvas[0:image_height, 0:image_width] = image

    # Define the position for the text
    text_x = image_width + 20
    text_y = (canvas_height + text_height) // 2

    # Put the text on the canvas
    cv2.putText(canvas, transcription, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

    return canvas

# Function to process images and their respective annotations from a given directory
def process_directory(directory, annotations_file):
    output_folder = os.path.join(output_master_folder, os.path.basename(directory))
    os.makedirs(output_folder, exist_ok=True)

    # Read the annotations file
    with open(annotations_file, 'r') as file:
        lines = file.readlines()

    # Process each line in the annotations file
    for line in tqdm(lines, desc=f"Processing images in {os.path.basename(directory)}"):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        image_path = parts[0]
        transcription = parts[1]

        # Create the full path to the image
        full_image_path = os.path.join(directory, os.path.basename(image_path))

        # Create the canvas with image and transcription
        canvas = create_canvas(full_image_path, transcription)
        if canvas is not None:
            # Save the canvas
            output_image_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_image_path, canvas)

# Process each subdirectory in the input master folder
subdirectories = [d for d in os.listdir(input_master_folder) if os.path.isdir(os.path.join(input_master_folder, d))]
for subdirectory in subdirectories:
    full_subdirectory_path = os.path.join(input_master_folder, subdirectory)
    annotations_file = os.path.join(input_master_folder, f"{subdirectory}.txt")
    if os.path.exists(full_subdirectory_path) and os.path.exists(annotations_file):
        process_directory(full_subdirectory_path, annotations_file)
    else:
        print(f"Directory {full_subdirectory_path} or file {annotations_file} does not exist.")

print("All images have been processed and saved in the recog_data_check folder.")
