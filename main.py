from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("C:/Users/suraj/Downloads/Model/saved_models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return f"Good Morning"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    
    img_batch = np.expand_dims(image, axis=0) # axis = 0 means its adds one more dimension horizontally & for 1 it adds one dimension vertically
    
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction[0])
    
    return {"predicted_class": predicted_class, "confidence": float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8070)

# ****************** ADVANTAGES OF FASTAPI *********************

# FastAPI offers inbuilt data validation
# in built documentation
#  Fast Running Performance
#  Less Time to Write code, few bugs
# https://f4c9-14-139-207-163.ngrok.io/predict