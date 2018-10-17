import numpy as np
import matplotlib.pyplot as plt
import os, argparse, math, random

''' 
	Select k distinct random vectors from the features dataset for each (class(combining all images data))
	Now, for each image we want to have a k(32) dimensional bag of visual words
'''

numOfClusters = 2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
args = vars(ap.parse_args())

cnt = 0;
wholeData = [];
lengthOfFile = [];

def fileHandle(fileName):
	file = open(fileName)
	tempList = []

	for line in file:
		teLine = line.rstrip('\n ').split(' ')
		nLine = [float(i) for i in teLine]
		wholeData.append(nLine)

	file.close()
	return

def euclidDist(centroid, dataPt):
	centroid = np.array(centroid)
	dataPt = np.array(dataPt)
	ldist = centroid-dataPt
	return np.sum(np.transpose(ldist)*ldist)

def distArray(dataPt):
	distVector = np.zeros((numOfClusters))
	for ind in range(numOfClusters):
		distVector[ind] = euclidDist(meanVector[ind], dataPt)
	return distVector

#assignment of clusters
def assignDataPt():
	costFunc = 0.0
	for dataPt in wholeData:
		distVector = distArray(dataPt)
		noCl = np.argmin(distVector)
		clusters[noCl].append(dataPt)
		costFunc += distVector[noCl]
	return costFunc

def findMean(component):
	total = np.zeros((len(component[0])))
	for point in component:
		total += np.array(point)
	total /= len(component)
	return total


def reCalcMean():
	for i in range(numOfClusters):
		meanVector[i] =  findMean(clusters[i])

print("Process start")
for root, dirs, files in os.walk(args["source"]):
	for f in files:
		path = os.path.relpath(os.path.join(root, f), ".")
		print("reading file: ",path)
		fileHandle(path)
		lengthOfFile.append(len(wholeData)-cnt)
		cnt = len(wholeData)


def allUnique(x):
	seen = list()
	return not any(i in seen or seen.append(i) for i in x)

while True:
	meanVector = random.sample(wholeData, numOfClusters)
	if allUnique(meanVector):
		break


#Now, k-means clustering
J = 0.0					#present Cost function
Jprev = -1.0			#previous Cost function
threshold = 1e-3

clusters = []
for i in range(numOfClusters):
	clusters.append([])

meanVector = np.array(meanVector)

counter = 0
while True:
	J = assignDataPt()
	print(counter," : J: ", J)
	counter += 1
	if Jprev != -1 and Jprev - J < threshold:
		break
	Jprev = J
	reCalcMean()
