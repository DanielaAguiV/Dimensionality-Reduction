import numpy as np
from PIL import Image
import os
from os import listdir

class Picture:
    def __init__(self,path) -> str:
        self.path = path

    def load_image(self):
        return Image.open(self.path)
    
    def edit_image(self):
        img = Image.open(self.path).resize((256,256)).convert('L')
        return img

    def image_asarray(self):
        img = Image.open(self.path).resize((256,256)).convert('L')
        numpydata = np.asarray(img)
        return numpydata


class PictureOperations:
    @staticmethod
    def image_fromarray(array):
        return Image.fromarray(array,'L')
    
    @staticmethod
    def dif_images(array_1,array_2):
        return np.mean(np.sum(np.array([array_1,-array_2]),axis = 0))
    