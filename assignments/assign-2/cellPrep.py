import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import errno


def create(source, imageName, path, outputFolder, targetPath):
    imagePath = path
    targetPath = outputFolder+"\\"+targetPath
    print("Image name: ", imageName)
    try:
        #print("CAT 1")
        image = mpimg.imread(imagePath)
        #print(image)
        print(len(image), " : ", len(image[0]))
        process(image, targetPath)
        if(len(image) % 32 != 0 or len(image[0]) % 32 != 0):
            print("NOT SCALED")
    except IOError:
        print("Not an Image File ", imageName)

def scale256(number):
    if(number < 1):
        return int(number*256/32)
    else:
        return int(number/32)

def scaleTo32(features, sum):
    for i in range(24):
        features[i] = int(features[i]*32*32/sum)
    return features


def processPatch(patch, featureVectors):
    # features = [0]*24  # list(24)
    # # print(patch)
    # # if len(patch) != 32 or len(patch[0]) !=32:
    # #     print("Bad dimensions")
    # #     print("Patch height: ", len(patch),"; patch breadth: ", len(patch[0]))
    # count = 0
    # for i in patch:
    #     for j in i:
    #         features[scale256(j[0])] += 1
    #         features[scale256(j[1])+8] += 1
    #         features[scale256(j[2])+16] += 1
    #         count += 1
    #         # print("something")
    # if(count < 32*32):
    #     features = scaleTo32(features, count)
    # # print("features: ",features)
    # featureVectors.append(features)
    # return featureVectors
    pass


def processImage(image, featureVectors):
    # height = len(image)
    # breadth = len(image[0])
    # patch = []
    # # print("Height: ",height, "; breadth: ",breadth)
    # for i in range(0, height, 32):
    #     # print(i,end=' ',flush=True)
    #     for j in range(0, breadth, 32):
    #         # print("i: ",i," j: ",j)
    #         if(height-i > 31 and breadth-j > 31):
    #             patch = image[i: (i+32), j: (j+32)]
    #             # print("image body")
    #         elif (height-i > 31):
    #             patch = image[i:(i+32), j:]
    #             # print("image side")
    #         elif (breadth-j > 31):
    #             patch = image[i:, j: (j+32)]
    #             # print("image bottom")
    #         else:
    #             patch = image[i:, j:]
    #             # print("image corner")
    #         # print(patch)
    #         featureVectors = processPatch(patch, featureVectors)
    return featureVectors


def process(image, targetPath):
    # featureVectors = []
    # # print("CAT")
    # featureVectors = processImage(image, featureVectors)
    # # print("CAT2")
    # if not os.path.exists(os.path.dirname(targetPath)):
    #     try:
    #         os.makedirs(os.path.dirname(targetPath))
    #     except OSError as exc:  # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise
    # try:
    #     print("target File: ", targetPath)
    #     outfile = open(targetPath, "w")
    # except IOError:
    #     print("File not created !!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("Output filename : ", targetPath)
    # # print("features: ")
    # for i in featureVectors:
    #     for j in i:
    #         outfile.write(str(j)+" ")
    #         # print(j,  end=' ', flush=True)
    #     outfile.write("\n")
    #     # print("\n")
    # outfile.close
    # print("YAAAAAAAAAAAAAAAAAY")
    return
