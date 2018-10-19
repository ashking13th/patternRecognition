import gmm 
import os
import kmeans
import numpy as np
import grapher
import argparse
import segmentation as sg
import cellProcess as cpr

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="Output data set location")
args = vars(ap.parse_args())

i = 0
for root, dirs, files in os.walk(args["source"]):
    for f in files:
        i += 1
        print("Image No. : ", i)
        path = os.path.relpath(os.path.join(root, f), ".")
        target = os.path.relpath(os.path.join(root, os.path.splitext(f)[0]))
        cpr.processFile(path, f, target)


def fileHandle(fileName):
    wholeData = []
    file = open(fileName)
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        nLine = np.array(nLine)
        wholeData.append(nLine)
    file.close()
    return wholeData
