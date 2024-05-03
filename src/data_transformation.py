import os
import zipfile
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.utils import load_img
from sklearn.model_selection import train_test_split

directory_to_extract_to = "/home/pranjal/Downloads/Building_footprint_segmentation/Data"

images = "/home/pranjal/Downloads/Building_footprint_segmentation/Data/segmentation_data/src"
masks = "/home/pranjal/Downloads/Building_footprint_segmentation/Data/segmentation_data/label"
image_height = 256
image_width = 256


class DataTransformation:

    def data_loader(self, path_to_zip_file):
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        #IMG_IDS = next(os.walk(images))[2]
        IMG_IDS = [file for file in os.listdir(images) if os.path.isfile(os.path.join(images, file))]


        X = np.zeros((len(IMG_IDS), image_height, image_width, 1), dtype=np.float32)
        y = np.zeros((len(IMG_IDS), image_height, image_width, 1), dtype=np.float32)

        for n, ids in enumerate(IMG_IDS):
            img = load_img(os.path.join(images, ids))
            x_img_arr = np.asarray((img))
            x_img = resize(x_img_arr, (image_height,image_width, 3), mode='constant', preserve_range=True)
            x_img = rgb2gray(x_img)
            x_img = np.expand_dims(x_img, axis=-1)

            mask  = load_img(os.path.join(masks, ids))
            y_mask_arr = np.asarray(mask)
            y_mask = resize(y_mask_arr, (image_height, image_width, 3), mode='constant', preserve_range=True)
            y_mask = rgb2gray(y_mask)
            y_mask = np.expand_dims(y_mask, axis=-1)

            X[n] = x_img/255.0
            y[n] = y_mask/255.0

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        return X_train, X_valid, y_train, y_valid


if __name__ == "__main__":
    X_train, X_valid, y_train, y_valid = DataTransformation().data_loader()
    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)