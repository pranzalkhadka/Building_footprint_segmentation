import albumentations as A
import os
from PIL import Image
import numpy as np

# Set the input and output folder paths
input_folder = '/home/pranjal/Downloads/ImageSegmentation/Satellite_dataset/image'
label_folder = '/home/pranjal/Downloads/ImageSegmentation/Satellite_dataset/label'
output_folder = '/home/pranjal/Downloads/ImageSegmentation/output'

# Create the output folder and subfolders if they don't exist
os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)

# Define the transformations to be applied
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
])

# Get the list of image files in the input folder
files = os.listdir(input_folder)

# Loop over the input images
for i, file in enumerate(files):
    print(f"Processing image {i+1}/{len(files)}: {file}")
    # Construct the new filenames with a prefix
    output_image_name = f"aug_{i:04d}.tif"
    output_annotation_name = f"aug_{i:04d}.tif"

    # Read the input image and annotation file
    image_path = os.path.join(input_folder, file)
    annotation_path = os.path.join(label_folder, file)
    image = Image.open(image_path)
    annotation = Image.open(annotation_path)

    # Apply the transformations
    transformed = transform(image=np.array(image), mask=np.array(annotation))
    transformed_image = transformed['image']
    transformed_annotation = transformed['mask']

    # Save the transformed image and annotation file with the new names
    output_image_path = os.path.join(output_folder, 'images', output_image_name)
    output_annotation_path = os.path.join(output_folder, 'annotations', output_annotation_name)
    Image.fromarray(transformed_image).save(output_image_path)
    Image.fromarray(transformed_annotation).save(output_annotation_path)
