import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

class imageToHistogram:
    imagePath = "/"
    targetPath = "/"
    image = 0
    featureVectors = []

    def __init__(self, source, imageName, path, outputFolder):
        imagePath = path
        targetPath = outputFolder+"/"+path
        #print("Image name: ",imageName)
        try:
            #print("CAT 1")
            image = mpimg.imread(imagePath)
            self.process()
            #print(image)
            # print(len(image)," : ", len(image[0]))
            # if(len(image) % 32 != 0 or len(image[0]) % 32 != 0):
            #     print("NOT SCALED")
        except IOError:
            print("Not an Image File")

    def processPatch(self, patch):
        if(len(patch) )

    def processImage(self):
        pass

    def process(self):

        pass
