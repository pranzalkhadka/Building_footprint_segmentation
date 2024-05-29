from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from keras.models import load_model
import tempfile
import matplotlib.pyplot as plt

app = FastAPI()


model = load_model('/home/pranjal/Downloads/Building_footprint_segmentation/models/model.h5')


def prediction(image):
    image_height, image_width = 256, 256
    test_image_resized = cv2.resize(image, (image_height, image_width))
    test_image_gray = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)
    test_image_normalized = test_image_gray / 255.0
    test_image_input = np.expand_dims(test_image_normalized, axis=-1)
    test_image_input = np.expand_dims(test_image_input, axis=0)

    predicted_mask = model.predict(test_image_input)
    predicted_mask = predicted_mask.reshape(image_height, image_width)
    predicted_mask_binary = np.where(predicted_mask > 0.5, 1, 0)
    return predicted_mask_binary


def plot_prediction(test_image_gray, predicted_mask_binary):
    plt.subplot(1, 2, 1)
    plt.imshow(test_image_gray, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask_binary, cmap='binary')
    plt.title('Predicted Mask')

    plt.show()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(await file.read())
        temp_image.seek(0)
        img = cv2.imread(temp_image.name)

    predicted_mask_binary = prediction(img)
    plot_prediction(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), predicted_mask_binary)

    return HTMLResponse(content="<h1>Prediction Completed</h1>", status_code=200)