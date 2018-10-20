import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--source", required=True, help="Raw data set location")
# ap.add_argument("-d", "--dest", required=True, help="Output data set location")
# args = vars(ap.parse_args())


segmentColors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (100, 0, 0),
          (0, 100, 0), (0, 0, 100), (200, 0, 0), (0, 200, 0),
          (0, 0, 200), (100, 200, 0), (200, 100, 0), (100, 0, 200),
          (200, 0, 100), (0, 100, 200), (0, 200, 100), (100, 100, 100),
          (200, 200, 200), (255, 255, 255), (0,0,0), (150, 150, 150),
          (50, 50, 50), (0, 50, 100), (0, 50, 200), (200, 0, 50),
          (0, 200, 50), (0, 50, 50), (100, 0, 100), (100, 100, 0),
          (150, 0, 200), (150, 200, 0),(200, 0, 150),(150, 255, 0),
          
          (255, 0, 255), (0, 255, 185), (255, 185, 0), (100, 230, 0),
          (185, 180, 0), (0, 0, 185), (200, 25, 0), (75, 200, 0),
          (185, 180, 200), (100, 200, 185), (75, 180, 0), (100, 230, 200),
          (200, 180, 100), (185, 100, 200), (230, 200, 100), (100, 100, 100),
          (200, 200, 200), (255, 255, 185), (0, 230, 0), (150, 230, 150),
          (50, 180, 50), (0, 50, 185), (0, 230, 200), (200, 230, 50),
          (180, 180, 50), (0, 50, 185), (100, 230, 100), (100, 230, 0),
          (150, 180, 200), (150, 200, 185), (200, 230, 150), (150, 230, 0)]


patchSize = 7


def gaussian(covMat, x, mean):
    # print(mean)
    numFeature = np.size(mean)
    gaussian = -(1/2)*np.sum((np.transpose(x-mean)
                              * (np.linalg.pinv(covMat)))*(x-mean))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    # print("Gaussian: ",gaussian)
    return gaussian


def gammaAllot(x, covMatVect, meanVector, piVect, clusters):
    gammaVect = np.zeros((clusters))
    sum = 0
    gaussians = np.zeros((clusters))
    # print("Gamma length: ", len(gammaVect))
    # print("Pi length: ", len(piVect))
    # print("Mean length: ", len(meanVector))

    for k in range(clusters):
        gaussians[k] = gaussian(covMatVect[k], x, meanVector[k])
    for k in range(clusters):
        sum += piVect[k]*gaussians[k]
    for k in range(clusters):
        gammaVect[k] = (piVect[k]*gaussians[k])/sum
    ans = np.argmax(gammaVect)
    # if ans != 0:
        # print("gamma vect: ",gammaVect)
        # print("Allotment : ", ans)
    return ans


def allotCluster(patchFeatures, meanVector):
    dist = []
    for mean in meanVector:
        dist.append(np.linalg.norm(patchFeatures-mean)**2)
    mini = np.argmin(dist)
    return mini


# def createOverlay(height, breadth, allotment):
#     overlay = []

#     for i in range(height):
#         row = []
#         for j in range(breadth):
#             row.append(segmentColors[allotment[int(i/7)][int(j/7)]])
#         overlay.append(row)
#     return overlay


# def process(image, features, meanVector):
#     height = len(image)
#     breadth = len(image[0])

#     allotment = []
#     patchCount = 0

#     # print("Height: ",height, "; breadth: ",breadth)
#     for i in range(0, height, 7):
#         # print(i,end=' ',flush=True)
#         row = []
#         for j in range(0, breadth, 7):
#             segment = allotCluster(features[patchCount], meanVector)
#             row.append(segment)
#             patchCount += 1
#         allotment.append(row)

#     overlay = createOverlay(height, breadth, allotment)
#     plt.figure()

#     plt.subplot(1,2,1)
#     plt.imshow(image, 'gray', interpolation='none')
#     plt.subplot(1,2,2)
#     plt.imshow(image, 'gray', interpolation='none')
#     plt.imshow(overlay, 'jet', interpolation='none', alpha=0.7)
#     plt.show()

def dummy(name):
    print("DUMMY")
    count = 0
    for c in colors:
        # img = Image.new('RGB',(512,512),color=c)
        fname = name+"\\"+str(count)+".jpg"
        # print("writing: ",name)
        # img.save(fname,"JPEG")
        count += 1


def cellSegmentaion(height, breadth, allotment, targetPath, dest, filename,clusters):
    im = Image.new('RGB', (breadth, height), color=(0, 0, 255))
    img = im.load()

    print("height: ",height,"; breadth: ",breadth)

    for i in range(height):
        for j in range(breadth):
            # print("j: ",j," ; i: ",i)
            a = allotment[int(i/patchSize)][int(j/patchSize)]
            # print("a: ",a)
            img[j, i] = (int(64*a/clusters), int(128*a/clusters), int(256*a/clusters))

    # img.draw()
    path = dest+"\\"+filename#+".jpg"
    # print("target path: ", path)
    im.save(path, "JPEG")
    # print("YAAY")


def segment(height, breadth, features, targetPath, filename, meanVector, dest, covMatVect, piVect, clusters):
    print("No of clusters: ",clusters)
    allotment = []
    abc = []
    patchCount = 0
    counter = np.zeros(clusters)

    # print("Height: ",height, "; breadth: ",breadth)
    for i in range(0, height, patchSize):
        # print(i,end=' ',flush=True)
        row = []
        for j in range(0, breadth, patchSize):
            # seg = allotCluster(features[patchCount], meanVector)
            seg = gammaAllot(features[patchCount], covMatVect, meanVector, piVect, clusters)
            counter[seg] += 1
            # print("Seg: ",seg)
            row.append(seg)
            abc.append(seg)
            patchCount += 1
        allotment.append(row)
    print("gamma distribution: ",counter)
    # print(allotment)
    print("abc len: ", len(abc))
    xFeature = []
    yFeature = []
    numOfClusters = len(meanVector)
    numOfPoints = len(features)
    print("Number of points: ", numOfPoints)
    print("A: ",len(allotment))
    print("B: ",len(allotment[0]))
    # for i in range(numOfClusters):
    #     xFeature.append([])
    #     yFeature.append([])

    # for i in range(numOfPoints):
    #     # print("abc[i]: ",abc[i])
    #     # print(yFeature[abc[i]])
    #     # print("features[i][0]: ",features[i], "\t ; i: ",i)
    #     # print("CAT: ",xFeature[abc[i]])
    #     xFeature[abc[i]].append(features[i][0])
    #     yFeature[abc[i]].append(features[i][1])

    # for i in range(numOfClusters):
    #     plt.scatter(xFeature[i], yFeature[i], marker='.', s=1)
    #     plt.scatter([meanVector[i][0]], [meanVector[i][1]], marker='*')

    cellSegmentaion(height, breadth, allotment, targetPath, dest, filename, clusters)
    return
