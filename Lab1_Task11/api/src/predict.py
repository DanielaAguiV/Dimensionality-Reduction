import os
import numpy as np
import tensorflow as tf
from PIL import Image


class Predict:

    def __init__(self, model):
        self.model = model

    def predict(self, image):

        #model = tf.keras.models.load_model(f'{self.model_folder}my_model.h5')
        number = Image.open(image).resize((28,28)).convert('L')
        number_array = np.asarray(number)
        number_array = number_array.reshape((1, 28, 28, 1))
        pred = self.model.predict(number_array)

        return pred.argmax()
