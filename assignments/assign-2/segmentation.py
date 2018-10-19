import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="Output data set location")
args = vars(ap.parse_args())


segmentColors = [[0,0,255],[0,255,0],[255,0,0]]
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

def allotCluster(patchFeatures, meanVector):
    allotment = []
    clusters = len(meanVector)

    for patch in patchFeatures:
        dist = []
        for mean in meanVector:
            dist.append(np.linalg.norm(patch-mean, axis=1)**2)
        allotment.append(np.argmin(dist))
    return allotment


def createOverlay(height, breadth, allotment):
    overlay = []

    for i in range(height):
        row = []
        for j in range(breadth):
            row.append(segmentColors[allotment[int(i/7),int(j/7)]])
        overlay.append(row)
    return overlay


def process(image, features, meanVector):
    height = len(image)
    breadth = len(image[0])

    allotment = []
    patchCount = 0

    # print("Height: ",height, "; breadth: ",breadth)
    for i in range(0, height, 7):
        # print(i,end=' ',flush=True)
        row = []
        for j in range(0, breadth, 7):
            segment = allotCluster(features[patchCount], meanVector)
            row.append(segment)
            patchCount += 1
        allotment.append(row)
    
    overlay = createOverlay(height, breadth, allotment)
    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(image, 'gray', interpolation='none')
    plt.subplot(1,2,2)
    plt.imshow(image, 'gray', interpolation='none')
    plt.imshow(overlay, 'jet', interpolation='none', alpha=0.7)
    plt.show()


def cellSegmentaion(height, breadth, allotment, targetPath):
    img = Image.new('RGB', (512,512), color=(0,0,255)).load()
    
    for i in range(height):
        for j in range(breadth):
            img[j,i] = colors[allotment[int(i/7), int(j/7)]]

    img.show()
    img.save(targetPath+".jpg")

def segment(features, targetPath, filename, meanVector):
    height = 512
    breadth = 512

    allotment = []
    patchCount = 0

    # print("Height: ",height, "; breadth: ",breadth)
    for i in range(0, height, 7):
        # print(i,end=' ',flush=True)
        row = []
        for j in range(0, breadth, 7):
            segment = allotCluster(features[patchCount], meanVector)
            row.append(segment)
            patchCount += 1
        allotment.append(row)

    cellSegmentaion(height, breadth, allotment, targetPath)
    return