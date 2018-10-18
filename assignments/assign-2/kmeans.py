import numpy as np
import matplotlib.pyplot as plt
import os, argparse, math, random
from datetime import datetime
from sklearn.cluster import KMeans

''' 
	Select k distinct random vectors from the features dataset for each (class(combining all images data))
	Now, for each image we want to have a k(32) dimensional bag of visual words
'''
start_time = datetime.now()

numOfClusters = 32

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
args = vars(ap.parse_args())

cnt = 0
wholeData = []
lengthOfFile = []

def fileHandle(fileName):
	file = open(fileName)
	for line in file:
		teLine = line.rstrip('\n ').split(' ')
		nLine = [float(i) for i in teLine]
		nLine = np.array(nLine)
		wholeData.append(nLine)

	file.close()
	return

def euclidDist(centroid, dataPt):
	# dataPt = np.array(dataPt)
	ldist = centroid-dataPt
	return np.sum(np.transpose(ldist)*ldist)

def distArray(dataPt):
	distVector = np.zeros((numOfClusters))
	for ind in range(numOfClusters):
		# norm = np.linalg.norm(meanVector[ind] - dataPt)
		distVector[ind] = euclidDist(meanVector[ind], dataPt)
	return distVector

#assignment of clusters
def assignDataPt():
	costFunc = 0.0
	for dataPt in wholeData:
		# print(type(dataPt))
		distVector = distArray(dataPt)
		noCl = np.argmin(distVector)	
		clusters[noCl].append(dataPt)
		costFunc += distVector[noCl]
	return costFunc

def findMean(component):
	component = np.array(component)
	total = np.zeros((len(component[0])))
	for point in component:
		total += point
	total /= len(component)
	return total


def reCalcMean():
	for i in range(numOfClusters):
		# print("cluser = ", clusters[i], ": i = ", i)
		# print("mean by np = ", np.mean(np.array(clusters[i]), axis=0))
		meanVector[i] = findMean(clusters[i])
		clusters[i] = []


# def allUnique(x):
# 	seen = list()
# 	return not any(i in seen or seen.append(i) for i in x)


print("Process start")
for root, dirs, files in os.walk(args["source"]):
	for f in files:
		path = os.path.relpath(os.path.join(root, f), ".")
		print("reading file: ",path)
		fileHandle(path)
		lengthOfFile.append(len(wholeData)-cnt)
		cnt = len(wholeData)



wholeData = np.array(wholeData)
meanVector = []
# while True:
# 	meanVector = random.sample(wholeData, numOfClusters)
# 	if allUnique(meanVector):
# 		break

def initMean():
	global meanVector
	temp = random.sample(range(len(wholeData)-1), numOfClusters)
	for assign in temp:
		meanVector.append(wholeData[assign])
	meanVector = np.array(meanVector)

#Now, k-means clustering
J = 0.0					#present Cost function
Jprev = -1.0			#previous Cost function
threshold = 1e-3

clusters = []
for i in range(numOfClusters):
	clusters.append([])

# meanVector = np.array(meanVector)

# kmeans = KMeans(n_clusters=32, init='random', n_init=10, max_iter=300, tol=0.001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='full').fit(wholeData)
# print(kmeans.n_iter_)
# print(kmeans.inertia_)

initMean()

counter = 0
while True:
	a = datetime.now()
	J = assignDataPt()
	# print(counter," : J: ", J, "\t : ",(Jprev-J)," : ",(datetime.now()-loopStarttime))
	print(counter, " : Cost = ", (Jprev-J))
	counter += 1
	if Jprev != -1 and Jprev - J < threshold:
		break
	Jprev = J
	reCalcMean()
	b = datetime.now()
	print(b-a)
