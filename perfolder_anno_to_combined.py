import os
import json
import shutil
from tqdm import tqdm

# Define the directories
root_folder = "/home/frinksserver/subhra/paddleOCR_data_preparation/separated"  # Root folder containing all subfolders with Label.txt
image_folder = "/home/frinksserver/subhra/paddleOCR_data_preparation/combined"
output_file = "combined_paddle.txt"
os.makedirs(image_folder, exist_ok=True)

def process_label_files(root_folder, image_folder, output_file):
    with open(output_file, 'w') as out_file:
        # Iterate over each subfolder in the root folder
        for subfolder in tqdm(os.listdir(root_folder), desc="Processing subfolders"):
            label_folder_path = os.path.join(root_folder, subfolder)
            label_file_path = os.path.join(label_folder_path, "Label.txt")
            
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as label_file:
                    for line in tqdm(label_file, desc="Processing images"):
                        image_name, annotations = line.strip().split('\t')
                        image_name = os.path.basename(image_name)
                        image_path = os.path.join(label_folder_path, image_name)
                        combined_image_path = os.path.join(image_folder, image_name)
                        
                        # Verify if the image exists in the original location
                        if os.path.exists(image_path):
                            annotations = json.loads(annotations)
                            
                            # Copy image to the combined folder
                            shutil.copy(image_path, combined_image_path)

                            # Write to output file with the updated path
                            out_file.write(f"{combined_image_path}\t{json.dumps(annotations)}\n")

# Run the function
process_label_files(root_folder, image_folder, output_file)

print(f"{output_file} has been created successfully.")
