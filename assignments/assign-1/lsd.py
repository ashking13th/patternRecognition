#use of re module for getting dimension of feature vector
import re, numpy as np
import matplotlib.pyplot as plt
import math

def plot(covMat):
	minMax = np.zeros((numFeature,numFeature))
	res = 200
	precision = 0.1
	bkgPointSize = 0.01
	dataPointSize = 0.5
	for i in range(numFeature):
		minMax[i, 0] = np.ceil(1.5*np.amin(data[:, :, 0]))
		minMax[i, 1] = np.ceil(1.5*np.amax(data[:, :, 1]))

	#x = np.linspace(1.2*minMax[0,0], 1.2*minMax[0,1], res)
	#y = np.linspace(1.2*minMax[1,0], 1.2*minMax[1,1], res)

	x = np.arange(int(1.2*minMax[0,0]), int(1.2*minMax[0,1]), precision)
	y = np.arange(int(1.2*minMax[1,0]), int(1.2*minMax[1,1]), precision)

	# x = np.arange(-40, 40, precision)
	# y = np.arange(-40, 40, precision)
	
	#X,Y = np.meshgrid(x,y)

	# pointClass = np.zeros((nClass, numSample, numFeature))
	# for i in range(nClass):
	# 	for j in range(numFeature):
	# 		pointClass[i,:,j] = data[i,:,j]

	for lc in range(1,5):
		greenX = []
		greenY = []
		yellowX = []
		yellowY = []
		blueX = []
		blueY = []
		# pointGX = pointGY = pointYX = pointYY = pointBX = pointBY = ([] for i in range(6))
		

		for i in x:
			for j in y:
				dataPt = np.array([i,j])
				classNum = tellClass(dataPt, covMat, False if lc==1 else True, False if lc==2 else True, False if lc==3 else True)
				if classNum == 0:
					greenX.append(i)
					greenY.append(j)
				if classNum == 1:
					yellowX.append(i)
					yellowY.append(j)
				if classNum == 2:
					blueX.append(i)
					blueY.append(j)

		plt.scatter(greenX, greenY, marker='o', c = "g", s=bkgPointSize)
		plt.scatter(yellowX, yellowY, marker='o', c = "y", s=bkgPointSize)
		plt.scatter(blueX, blueY, marker='o', c = "b", s=bkgPointSize)

		if lc==1:
			plt.scatter(data[1,:,0],data[1,:,1], c = "r", s=dataPointSize)
			plt.scatter(data[2,:,0],data[2,:,1], c = "r", s=dataPointSize) 
		elif lc==2:
			plt.scatter(data[0,:,0],data[0,:,1], c = "r", s=dataPointSize)
			plt.scatter(data[2,:,0],data[2,:,1], c = "r", s=dataPointSize) 
		elif lc==3:
			plt.scatter(data[0,:,0],data[0,:,1], c = "r", s=dataPointSize)
			plt.scatter(data[1,:,0],data[1,:,1], c = "r", s=dataPointSize)
		else:
			plt.scatter(data[0,:,0],data[0,:,1], c = "r", s=dataPointSize)
			plt.scatter(data[1,:,0],data[1,:,1], c = "r", s=dataPointSize)
			plt.scatter(data[2,:,0],data[2,:,1], c = "r", s=dataPointSize) 


		plt.show()


#defining discriminant function


def discriminant(dataPt, mean, covariance):
	covInv = np.linalg.inv(covariance)
	dataTranp = np.transpose(dataPt)
	meanTranp = np.transpose(mean)

	W_i = np.multiply(covInv, -0.5)
	w_i = np.matmul(covInv, mean)
	bias_i = np.matmul(meanTranp,covInv)
	bias_i = np.matmul(bias_i,mean)
	bias_i += np.log(np.linalg.det(covariance))
	bias_i = np.multiply(bias_i, -0.5)
	Wtot_i = np.matmul(dataTranp,W_i)
	Wtot_i = np.matmul(Wtot_i,dataPt)
	w_i = np.transpose(w_i);
	w_i = np.matmul(w_i,dataPt)
	total = Wtot_i + w_i + bias_i
	return total

def tellClass(dataPt, covar, class1, class2, class3):
	#print(dataPt)
	discValueC = np.zeros((nClass))
	#print(discValueC)
	for k in range(nClass):
		discValueC[k] = discriminant(dataPt, meanVector[k], covar[k])
		
	if class1 & class2 & class3:
		return np.argsort(discValueC)[-1]
	if not class1:
		if discValueC[1] > discValueC[2]:
			return 1
		return 2
	if not class2:
		if discValueC[0] > discValueC[2]:
			return 0
		return 2
	if not class3:
		if discValueC[0] > discValueC[1]:
			return 0
		return 1


nClass = 3

file = open("class1.txt")
li = [];

for line in file:
	ali = line.rstrip('\n').split(' ')
	li.append(ali)

file.close()
numSample = len(li)
numFeature = len(li[0])

data = np.zeros((nClass, numSample, numFeature))
data[0] = np.array(li)
data[1] = np.loadtxt("class2.txt")
data[2] = np.loadtxt("class3.txt")

#meanVectors
#Can also use in-built function of numpy -- numpy.mean

meanVector = np.zeros((nClass, numFeature, 1))
for i in range(nClass):
	meanVector[i] = (np.sum(data[i],axis=0)).reshape((numFeature,1))
	meanVector[i] /= numSample

# Covariance matrix
# can use inbuilt cov func too

covMatrix = np.zeros((nClass,numFeature,numFeature))
for i in range(nClass):
	for j in range(numFeature):
		for k in range(numFeature):
			feature1 = data[:, j]
			feature2 = data[:, k]
			np.subtract(feature1, meanVector[i, j, 0])
			np.subtract(feature2, meanVector[i, k, 0])
			covMatrix[i] = np.matmul(np.transpose(feature1),feature2)
			covMatrix[i] /= (numSample-1)


# # now plotting points
# # plt.plot(data[:,0], data[:,1], 'ro')
# # plt.show()


#Classifier - 1
covMatrixCfier1 = np.zeros((numFeature, numFeature))

for i in range(nClass):
	for j in range(numFeature):
		for k in range(numFeature):
			if j!=k:
				covMatrixCfier1[j, k] = 0
			else:
				covMatrixCfier1[j, k] += covMatrix[i, j, k]

covMatrixCfier1 /= nClass

avgDiag = 0

for j in range(numFeature):
	for k in range(numFeature):
		if j==k:
			avgDiag += covMatrixCfier1[j, k]

avgDiag /= numFeature

for i in range(numFeature):
	for j in range(numFeature):
		if i==j:
			covMatrixCfier1[i,j] = avgDiag	

temp = np.copy(covMatrixCfier1)
covMatrixCfier1 = np.copy(covMatrix)

for i in range(nClass):
	covMatrixCfier1[i] = temp
##end of Classifier - 1

#Classifier - 2
covMatrixCfier2 = np.zeros((numFeature, numFeature))

for i in range(nClass):
	for j in range(numFeature):
		for k in range(numFeature):
			covMatrixCfier2[j, k] += covMatrix[i, j, k]

covMatrixCfier2 /= nClass

temp = np.copy(covMatrixCfier2)
covMatrixCfier2 = np.copy(covMatrix)

for i in range(nClass):
	covMatrixCfier2[i] = temp

## end classifier-2

#Classifier - 3
covMatrixCfier3 = np.copy(covMatrix)

for i in range(nClass):
	for j in range(numFeature):
		for k in range(numFeature):
			if j!=k:
				covMatrixCfier3[i, j, k] = 0

##end of classifier - 3

#for classifier - 4, use original matrix covMatrix

file = open("classt1.txt")
li = [];

for line in file:
	ali = line.rstrip('\n').split(' ')
	li.append(ali)

file.close()
testSample = len(li)

testData = np.zeros((nClass, testSample, numFeature))
testData[0] = np.array(li)
testData[1] = np.loadtxt("classt2.txt")
testData[2] = np.loadtxt("classt3.txt")



#classifier - 1
predictClassClf1 = np.zeros((nClass,nClass))

for i in range(nClass):
	for j in range(testSample):
		discValue = np.zeros(nClass)
		for k in range(nClass):
			discValue[k] = discriminant(testData[i][j],meanVector[k],covMatrixCfier1[k])
			predictClassClf1[i][np.argsort(discValue)[-nClass]] += 1

#end -- classifier - 1

#classifier - 2
predictClassClf2 = np.zeros((nClass,nClass))

for i in range(nClass):
	for j in range(testSample):
		discValue = np.zeros(nClass)
		for k in range(nClass):
			discValue[k] = discriminant(testData[i][j],meanVector[k],covMatrixCfier2[k])
			predictClassClf2[i][np.argsort(discValue)[-nClass]] += 1

#end -- classifier - - 2

#classifier - 3
predictClassClf3 = np.zeros((nClass,nClass))

for i in range(nClass):
	for j in range(testSample):
		discValue = np.zeros(nClass)
		for k in range(nClass):
			discValue[k] = discriminant(testData[i][j],meanVector[k],covMatrixCfier3[k])
			predictClassClf3[i][np.argsort(discValue)[-nClass]] += 1
#end Classifier-3

#classifier - 4
predictClassClf4 = np.zeros((nClass,nClass))

for i in range(nClass):
	for j in range(testSample):
		discValue = np.zeros(nClass)
		for k in range(nClass):
			discValue[k] = discriminant(testData[i][j],meanVector[k],covMatrix[k])
			predictClassClf4[i][np.argsort(discValue)[-nClass]] += 1

#end -- Classifier - 4
#plot(covMatrixCfier1)
#plot(covMatrixCfier2)
#plot(covMatrixCfier3)
plot(covMatrix)