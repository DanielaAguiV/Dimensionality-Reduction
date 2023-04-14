import os
from fastapi import FastAPI, File, UploadFile
import shutil
from src.utils import Utils
from src.predict import Predict
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

# Load the model in format h5
model = tf.keras.models.load_model(f'{os.getenv("MODEL_FOLDER")}{os.getenv("MODEL")}')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    upload_dir = os.path.join(os.getcwd(), os.getenv("IMAGE_FOLDER"))
    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path
    dest = os.path.join(upload_dir, file.filename)
    print(dest)

    # copy the file contents
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    predict = Predict(model)
    prediction = int(predict.predict(dest))
    utl = Utils()
    utl.delete_image(dest)

    return {"filename": file.filename, "number": prediction}
