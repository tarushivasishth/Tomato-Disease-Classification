from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",  # For your frontend URL
    "http://localhost:3000",  # If you're using React development server
    "http://127.0.0.1:5500",  # Allow static HTML from port 5500
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

MODEL = tf.keras.models.load_model(r"C:\Users\tarus\deep_learning_codebasics\tomato_classification\models\3.h5")

# Class names for prediction
CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
    'Septoria_leaf_spot', 'Spider_mites_Two_spotted_spider_mite', 
    'Target_Spot', 'Tomato_YellowLeaf__Curl_Virus', 
    'Tomato_mosaic_virus', 'healthy'
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run(app, host="localhost", port=5000)
