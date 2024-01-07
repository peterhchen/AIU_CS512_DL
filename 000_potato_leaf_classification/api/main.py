
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# ->: always return "np.ndarray" to prevent unexpected.
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # use "localhost:8000/docs" or "postman" to verigy 
    # fastAPI get and post.
    # await is asynchronous: read the file in the background
    # for all users and process the next users without waiting.
    #print("read_file_as_image:")
    #print('file:', file) 
    # file: <starlette.datastructures.UploadFile object at 0x7fa8eff51c30>
    image = read_file_as_image(await file.read())
    #print('image.shape:', image.shape)    # image.shape: (256, 256, 3)
    #print('image[0][0]:', image[0][0])    # image[0]: [108 106 120]

    img_batch = np.expand_dims(image, 0)
    #print('imag_batch.shape:', img_batch.shape) # imag_batch.shape: (1, 256, 256, 3)
    #print('MODEL:', MODEL)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    print('predicted_class:', predicted_class)
    confidence = np.max(predictions[0])
    print('confidence:', confidence)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
