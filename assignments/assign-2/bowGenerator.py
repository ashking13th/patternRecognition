import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
from datetime import datetime
# from sklearn.mixture import GaussianMixture as gm

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-b", "--bow", required=True, help="Raw BoVW data set location")
ap.add_argument("-o", "--output", required=True, help="output location")
args = vars(ap.parse_args())

def plotHistogram(values, bins, title, name):
    graphName = "Bag of Visual Words representation"
    # print("making graph: ")
    name = name+".jpg"
    plt.hist(values, bins, facecolor='green', rwidth=0.8)
    # print("histogram made: ")
    plt.xlabel("Visual Words (Clusters)")
    plt.ylabel("Number of patches")
    plt.title(graphName)
    # plt.show()
    print("plot path: ",name)
    plt.savefig(name)

def fileHandle(fileName):
    wholeData = []
    file = open(fileName)
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [int(float(i)) for i in teLine]
        nLine = np.array(nLine)
        # print(nLine)
        wholeData.append(nLine)
    file.close()
    return wholeData

i = 0
wholeData = fileHandle(args['bow'])
# print(wholeData)
fileCount = len(wholeData)
xvalues = range(1,33)

for root, dirs, files in os.walk(args["source"]):
    for f in files:
        i += 1
        # print("Image No. : ", i,  " ; ",f)
        path = os.path.relpath(os.path.join(root, f), ".")
        target = os.path.relpath(os.path.join(root, os.path.splitext(f)[0]))
        # print("target ", os.path.splitext(f)[0])
        # for i in range(fileCount):
        plotHistogram(wholeData[i], xvalues, target, args['output']+os.path.splitext(f)[0])


