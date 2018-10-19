# import gmm
import os
# import kmeans
import numpy as np
import grapher
import argparse
import segmentation as sg

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="Destination location")
ap.add_argument("-m", "--mean", required=True, help="mean file location")
ap.add_argument("-c", "--cov", required=True, help="covaMat file location")
ap.add_argument("-p", "--pi", required=True, help="piVect file location")

args = vars(ap.parse_args())


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

def covaMat(mat):
    vect = np.zeros(shape=(3,2,2))
    for i in range(3):
        for j in range(2):
            vect[i,j,j] = mat[i][j]
    return vect


meanVector = fileHandle(args['mean'])
covMatVect = covaMat(fileHandle(args['cov']))
piVect = fileHandle(args['pi'])
print(meanVector)
i = 0
for root, dirs, files in os.walk(args["source"]):
    for f in files:
        print("File: ", f)
        i += 1
        print("Image No. : ", i)
        path = os.path.relpath(os.path.join(root, f), ".")
        target = os.path.relpath(os.path.join(root, os.path.splitext(f)[0]))
        sg.segment(fileHandle(path), target, f, meanVector, args['dest'])
