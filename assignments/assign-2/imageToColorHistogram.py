import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

class imageToColorHistogram:
    imagePath = "/"
    targetPath = "/"
    image = 0

    def __init__(self, imageLoc, outputFolder):
        imagePath = imageLoc
        targetPath = outputFolder+"/"+imageLoc
        image = mpimg.imread(imagePath)
        print(image)

    # def processPath():
    #     pass

    # def processImage():
    #     pass