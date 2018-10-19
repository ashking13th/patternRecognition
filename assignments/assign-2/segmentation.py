import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="Output data set location")
args = vars(ap.parse_args())


segmentColors = [[0,0,255],[0,255,0],[255,0,0]]

def allotSegment(patchFeatures):
    return 1


def createOverlay(height, breadth, allotment):
    overlay = []

    for i in range(height):
        row = []
        for j in range(breadth):
            row.append(segmentColors[allotment[int(i/7),int(j/7)]])
        overlay.append(row)
    return overlay


def process(image, features):
    height = len(image)
    breadth = len(image[0])

    allotment = []
    patchCount = 0

    # print("Height: ",height, "; breadth: ",breadth)
    for i in range(0, height, 7):
        # print(i,end=' ',flush=True)
        row = []
        for j in range(0, breadth, 7):
            segment = allotSegment(features[patchCount])
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


def cellSegmentaion(height, breadth, clusterAssignments):
    pass

