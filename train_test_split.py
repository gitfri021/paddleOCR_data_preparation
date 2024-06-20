import os
import random
import shutil
from tqdm import tqdm

# Define the input file and output directories
input_file = "combined_paddle.txt"
output_dir = "/home/frinksserver/subhra/paddleOCR_data_preparation/split_det_text"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Create the output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def split_data(input_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Shuffle the data
    random.shuffle(lines)

    # Calculate the split indices
    total = len(lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split the data
    train_data = lines[:train_end]
    val_data = lines[train_end:val_end] if val_ratio > 0 else []
    test_data = lines[val_end:]

    # Write the split data to separate files
    write_split_data(train_data, train_dir, os.path.join(output_dir, "train.txt"))
    if val_data:
        val_dir = os.path.join(output_dir, "val")
        os.makedirs(val_dir, exist_ok=True)
        write_split_data(val_data, val_dir, os.path.join(output_dir, "val.txt"))
    write_split_data(test_data, test_dir, os.path.join(output_dir, "test.txt"))

def write_split_data(data, target_dir, output_filepath):
    with open(output_filepath, 'w') as f:
        for line in tqdm(data, desc=f"Writing data to {os.path.basename(output_filepath)}"):
            image_path, annotations = line.strip().split('\t')
            image_name = os.path.basename(image_path)
            target_image_path = os.path.join(target_dir, image_name)
            
            # Copy the image to the target directory
            shutil.copy(image_path, target_image_path)
            
            # Write the new image path and annotations to the output file
            f.write(f"{target_image_path}\t{annotations}\n")

# Example usage:
# To split into 80% train, 10% val, 10% test
split_data(input_file, train_ratio=0.8, val_ratio=0, test_ratio=0.2)

# To split into 90% train, 0% val, 10% test
# split_data(input_file, train_ratio=0.9, val_ratio=0, test_ratio=0.1)

print("Data has been split successfully.")
