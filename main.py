from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model1 = load_model('/home/pranjal/Downloads/Building_footprint_segmentation/models/model5.h5')
# model1.summary()


# import tensorflow as tf

# model2 = tf.keras.models.load_model('/home/pranjal/Downloads/Building_footprint_segmentation/models/model6.keras')
# model2.summary()

# def plot_prediction(test_image_gray, predicted_mask_binary):
#     # Plot the original image
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(test_image_gray, cmap='gray')
#     plt.title('Original Image')

#     # Plot the predicted mask
#     plt.subplot(1, 2, 2)
#     plt.imshow(predicted_mask_binary, cmap='binary')
#     plt.title('Predicted Mask')

#     plt.show()

def prediction_plot(predicted_mask_binary):

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask_binary, cmap='binary')
    plt.title('Predicted Mask')

    plt.show()


image_path = '/home/pranjal/Downloads/Building_footprint_segmentation/Data/segmentation_data/src/S008.png'
img = cv2.imread(image_path)

image_height, image_width = 256, 256
test_image_resized = cv2.resize(img, (image_height, image_width))
test_image_gray = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)
test_image_normalized = test_image_gray / 255.0
test_image_input = np.expand_dims(test_image_normalized, axis=-1)
test_image_input = np.expand_dims(test_image_input, axis=0)

predicted_mask = model1.predict(test_image_input)
predicted_mask = predicted_mask.reshape(image_height, image_width)
predicted_mask_binary = np.where(predicted_mask > 0.5, 1, 0)
prediction_plot(predicted_mask_binary)