# import gmm
import os
# import kmeans
import numpy as np
import grapher
import argparse
import imageSegmentation as sg
import matplotlib.image as mpimg

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
# ap.add_argument("-d", "--dest", required=True, help="Destination location")
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

def covaMat(mat, clusters, dimensions):
    vect = np.zeros(shape=(clusters, dimensions,dimensions))
    for i in range(clusters):
        for j in range(dimensions):
            vect[i,j,j] = mat[i][j]
    return vect


meanVector = fileHandle(args['mean'])
piVect = fileHandle(args['pi'])[0]
# print("Length of piVector: ",len(piVect))
clusters = len(meanVector)
dimensions = len(meanVector[0])
covMatVect = covaMat(fileHandle(args['cov']),clusters, dimensions)
print("Length of covMatVector: ", len(covMatVect))
# print(meanVector)
# for i in covMatVect:
#     print("matrix: ")
#     for j in i:
#         for k in j:
#             print(k,end=' ')
#         print("\n")
# print(piVect)

# sg.dummy(args['dest'])
i = 0
for root, dirs, files in os.walk(args["source"]):
    for f in files:
        print("File: ", f)
        i += 1
        # print("Image No. : ", i)
        path = os.path.relpath(os.path.join(root, f), ".")
        # print("target: ",path)
        height = 0
        breadth = 0
        try:
            #print("CAT 1")
            image = mpimg.imread(path)
            print(len(image), " : ", len(image[0]))
            height = len(image)
            breadth = len(image[0])
            # image.close()
        except IOError:
            print("Not an Image File ", f)

        target = os.path.relpath(os.path.join(root, f))
        dataTarget = "cellNonOverlap\\"+os.path.relpath(os.path.join(root, os.path.splitext(f)[0]))
        # print("dataTarget: ",dataTarget)
        sg.segment(height, breadth, fileHandle(dataTarget), target, f, meanVector, args['dest'],covMatVect, piVect, clusters)
