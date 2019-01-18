import numpy as np
import kmeans
import gmm
nCluster=1
nIteration=100
nClass=3

def fileHandle(fileName):
    file = open(fileName)
    tempList = []

    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        tempList.append(teLine)

    file.close()
    x = np.array(tempList,float)
    return x
def Gaussian(covMat, x, mean):
    # print(mean)
    numFeature = np.size(mean)
    gaussian = -(1.0/2)*np.sum((np.transpose(x-mean)*(np.linalg.inv(covMat)))*(x-mean))
    # gaussian = (-1/2)*(np.dot(np.transpose(x-mean), np.dot(np.linalg.inv(covMat), x-mean)))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    return gaussian
    # print("Gaussian: ", gaussian)

mainList = []
mainList.append(fileHandle("bayou_train.txt"))
mainList.append(fileHandle("chalet_train.txt"))
mainList.append(fileHandle("creek_train.txt"))

testList = []
testList.append(fileHandle("bayou_test.txt"))
testList.append(fileHandle("chalet_test.txt"))
testList.append(fileHandle("creek_test.txt"))

numSample = np.zeros((nClass))
numFeature = len(mainList[0][0])
for i in range(nClass):
    numSample[i] = len(mainList[i])

#meanVectors
#Can also use in-built function of numpy -- numpy.mean

meanVector = np.zeros((nClass, numFeature, 1))
for i in range(nClass):
    data = np.copy(mainList[i])
    meanVector[i] = (np.sum(data,axis=0)).reshape((numFeature,1))
    meanVector[i] /= numSample[i]

# Covariance matrix
# can use inbuilt cov func too
# covMatrix = np.zeros((nClass,numFeature,numFeature))
# scatterMatrix = np.zeros((nClass,numFeature,numFeature))
scatterMatrix = np.zeros((nClass,numFeature,numFeature))
for i in range(nClass):
    extData = np.copy(mainList[i])
    for j in range(numFeature):
        extData[:, j] = np.subtract(extData[:, j], meanVector[i, j, 0])
    scatterMatrix[i] = np.matmul(np.transpose(extData), extData)
    # scatterMatrix[i]=covMatrix[i]
    


directions = []
projectedData=[]
def caldirections(scatterMatrix,meanVector):
    directions1=[]
    for i in range(nClass):
        for j in range(i+1,nClass):
            dr=np.dot(np.linalg.inv(scatterMatrix[i]+scatterMatrix[j]),meanVector[i]-meanVector[j]).reshape(numFeature,1)
            p=np.sqrt(np.sum(np.square(dr)))
            dr/=p
            directions1.append(dr)
    return directions1

# comment p
directions=caldirections(scatterMatrix,meanVector)
countlda=0

for i in range(nClass):
    for j in range(i+1,nClass):
        projectedData1=[]
        projectedData1.append(np.matmul(mainList[i],directions[countlda]))
        projectedData1.append(np.matmul(mainList[j],directions[countlda]))
        projectedData.append(projectedData1)
        countlda+=1

parameters=[]
cnt=0
for i in range(nClass):
	for j in range(i+1,nClass):
		parameter=[]
		p=kmeans.kmeanIntialize(projectedData[cnt][0],nCluster,nIteration)
		p1=gmm.applyGMM(nCluster,projectedData[cnt][0],p[0],p[1])
		parameter.append(p1)
		p=kmeans.kmeanIntialize(projectedData[cnt][1],nCluster,nIteration)
		p1=gmm.applyGMM(nCluster,projectedData[cnt][1],p[0],p[1])
		parameter.append(p1)
		parameters.append(parameter)
		cnt+=1

confusionMatrix=np.zeros((nClass,nClass),int)
for i in range(nClass):
	for j in range(len(testList[i])):
		countlda=0
		lst=np.zeros(nClass)
		for k in range(nClass):
			for l in range(k+1,nClass):
				proDataPoint=np.matmul(testList[i][j],directions[countlda]).reshape(1,1)
				# p=discriminant(proDataPoint,projectedDataMean[countlda][0],projectedDataVar[countlda][0])
				# q=discriminant(proDataPoint,projectedDataMean[countlda][1],projectedDataVar[countlda][1])
				p=0
				q=0
				for m in range(nCluster):
					p+=parameters[countlda][0][0][m]*Gaussian(parameters[countlda][0][2][m],proDataPoint,parameters[countlda][0][1][m])
				for m in range(nCluster):
					q+=parameters[countlda][1][0][m]*Gaussian(parameters[countlda][1][2][m],proDataPoint,parameters[countlda][1][1][m])
				if(p>q):
					lst[k]+=1
				else:
					lst[l]+=1
				countlda+=1
		confusionMatrix[i][np.argmax(lst)]+=1



print(confusionMatrix)