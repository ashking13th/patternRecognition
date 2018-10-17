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
        print(len(image), " : ", len(image[0]))
        print(image.shape)
        # print(image)
        process(image, targetPath)
        if(len(image) % 32 != 0 or len(image[0]) % 32 != 0):
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$ NOT SCALED $$$$$$$$$$$$$$$$$$$$$$$$$")
    except IOError:
        print("Not an Image File ", imageName)

def scale256(number):
    if(number < 1):
        return int(number*256/32)
    else:
        return int(number/32)

def scaleTo(features, sum, scalefactor):
    for i in range(len(features)):
        features[i] = int(features[i]*scalefactor/sum)
    return features


def processPatch(patch, featureVectors):
    features = [] # [0]*24  # list(24)
    # print(patch)
    # if len(patch) != 32 or len(patch[0]) !=32:
    #     print("Bad dimensions")
    #     print("Patch height: ", len(patch),"; patch breadth: ", len(patch[0]))
    count = 0
    values = []
    for i in patch:
        for j in i:
            values.append(j)
            count += 1
            # print("something")
    # if(count < 49):
    #     values = scaleTo(features, count,49)
    nValues = np.array(values)
    features.append(np.mean(nValues, dtype=np.float64))
    features.append(np.var(nValues, dtype=np.float64))
    # print("features: ",features)
    featureVectors.append(features)
    return featureVectors


def processImage(image, featureVectors):
    height = len(image)
    breadth = len(image[0])
    patch = []
    i = 0
    # print("Height: ",height, "; breadth: ",breadth)
    for i in range(0, height, 1):
        # print(i,end=' ',flush=True)
        j = 0
        for j in range(0, breadth, 1):
            # print("i: ",i," j: ",j)
            if(height-i > 6 and breadth-j > 6):
                patch = image[i: (i+7), j: (j+7)]
                # print("image body")
            elif (height-i > 6):
                patch = image[i:(i+7), j:]
                # print("image side")
            elif (breadth-j > 6):
                patch = image[i:, j: (j+7)]
                # print("image bottom")
            else:
                patch = image[i:, j:]
                # print("image corner")
            # print(patch)
            featureVectors = processPatch(patch, featureVectors)
    return featureVectors


def process(image, targetPath):
    featureVectors = []
    # print("CAT")
    featureVectors = processImage(image, featureVectors)
    # print("CAT2")
    if not os.path.exists(os.path.dirname(targetPath)):
        try:
            os.makedirs(os.path.dirname(targetPath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    try:
        print("target File: ", targetPath)
        outfile = open(targetPath, "w")
    except IOError:
        print("File not created !!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Output filename : ", targetPath)
    print("features: ")
    n = 0
    for i in featureVectors:
        print("No.: ",n,  end=' ', flush=True)
        n += 1
        for j in i:
            outfile.write(str(j)+" ")
            print(j,  end=' ', flush=True)
        outfile.write("\n")
        print("\n")
    outfile.close
    print("YAAAAAAAAAAAAAAAAAY")
    return
