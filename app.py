import io
import PIL.ImageOps
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import PIL
from keras.models import load_model
from fastapi.responses import StreamingResponse


app = FastAPI()

model = load_model('/home/pranjal/Downloads/Building_footprint_segmentation/models/model5.h5')

image_height, image_width = 256, 256

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert("L")
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((image_height, image_width))
    img_array = np.array(pil_image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    predicted_mask = model.predict(img_array)
    predicted_mask = predicted_mask.reshape(image_height, image_width)
    predicted_mask_binary = np.where(predicted_mask > 0.5, 1, 0).astype(np.uint8)
    mask_image = PIL.Image.fromarray(predicted_mask_binary * 255)
    
    buf = io.BytesIO()
    mask_image.save(buf, format='PNG')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")



# @app.post("/predict-image/")
# async def predict_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     pil_image = PIL.Image.open(io.BytesIO(contents)).convert("L")
#     pil_image = PIL.ImageOps.invert(pil_image)
#     pil_image = pil_image.resize((image_height, image_width))
#     img_array = np.array(pil_image) / 255.0
#     img_array = np.expand_dims(img_array, axis=-1)
#     img_array = np.expand_dims(img_array, axis=0) 
#     predicted_mask = model.predict(img_array)
#     predicted_mask = predicted_mask.reshape(image_height, image_width)
#     predicted_mask_binary = np.where(predicted_mask > 0.5, 1, 0).tolist()

#     return {"without_prediction": predicted_mask_binary}