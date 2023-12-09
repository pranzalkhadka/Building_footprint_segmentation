import os

folder_path = '/home/pranjal/Downloads/ImageSegmentation/building_footprint/images'
old_extension = '.tiff' 
new_extension = '.tif'  

for filename in os.listdir(folder_path):
    if filename.endswith(old_extension):
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, filename.replace(old_extension, new_extension))
        os.rename(old_path, new_path)
