import os

class Utils:

    @staticmethod
    def delete_image(filename):
        os.remove(filename)
