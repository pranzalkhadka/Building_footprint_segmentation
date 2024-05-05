from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import keras
import numpy as np
import io
from fastapi.responses import StreamingResponse
import tensorflow as tf
import PIL
import cv2

image_height = 256
image_width = 256

model = tf.keras.models.load_model('/home/pranjal/Downloads/Building_footprint_segmentation/model1.keras')

app = FastAPI()

@app.get("/")
async def process_file():
    return {"filename": "file"}


@app.post("/predict")
async def process_image(file: UploadFile = File(...)):
    content = await file.read()
    img = cv2.imread(content)
    test_image_resized = cv2.resize(img, (image_height, image_width))
    test_image_gray = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)
    test_image_normalized = test_image_gray / 255.0
    test_image_input = np.expand_dims(test_image_normalized, axis=-1)
    test_image_input = np.expand_dims(test_image_input, axis=0)

    predicted_mask = model.predict(test_image_input)

    predicted_mask = predicted_mask.reshape(image_height, image_width)
    predicted_mask_binary = np.where(predicted_mask > 0.5, 1, 0)

    return StreamingResponse(io.BytesIO(predicted_mask_binary), media_type="image/png")




# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import keras
# import io
# from fastapi.responses import StreamingResponse
# import tensorflow as tf
# import PIL

# image_height = 256
# image_width = 256

# model = tf.keras.models.load_model('/home/pranjal/Downloads/Building_footprint_segmentation/model1.keras')

# app = FastAPI()

# @app.get("/")
# async def process_file():
#     return {"filename": "file"}



# @app.post("/predict")
# async def process_image(file: UploadFile = File(...)):
#     # Your image processing logic here
#     return {"filename": file.filename}

