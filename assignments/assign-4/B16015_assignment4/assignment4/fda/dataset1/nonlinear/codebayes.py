import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

nClass = 3
totalFda=nClass*(nClass-1)/2

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

def plot():
	# for i in range(totalFda):
	colors = ['#136906', '#fcbdfc', '#e5ff00', '#ff0000', '#3700ff', '#000000']
	plt.scatter(mainList[0][:,0],mainList[0][:,1],marker='.',color=colors[4])
	plt.scatter(mainList[1][:,0],mainList[1][:,1],marker='.',color=colors[5])
	plotname = "Class12data.png"
	plt.title("Class 1 and Class 2")
	plt.xlabel('X')
	plt.ylabel('Y')
	class_colours = [colors[4], colors[5]]
	classes = ["Class 1 Data", "Class 2 Data"]
	recs = []
	for i in range(0,len(class_colours)):
		recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
	plt.legend(recs, classes, loc='upper right')
	plt.savefig(plotname)
	plt.show()
	plt.scatter(projectedData[0][0],np.zeros(len(projectedData[0][0]),float),marker='.',color=colors[4])
	plt.scatter(projectedData[0][1],np.zeros(len(projectedData[0][1]),float),marker='.',color=colors[5])
	plotname = "Class12reducedata.png"
	plt.title("Class 1 and Class 2")
	plt.xlabel('X')
	plt.ylabel('Y')
	class_colours = [colors[4], colors[5]]
	classes = ["Class1 reduce Data", "Class2 reduce Data"]
	recs = []
	for i in range(0,len(class_colours)):
		recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
	plt.legend(recs, classes, loc='upper right')
	plt.savefig(plotname)
	plt.show()

def discriminant(dataPt, mean, covariance):
	# print(dataPt.shape,mean.shape,covariance.shape)
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

mainList = []
mainList.append(fileHandle("class1.txt"))
mainList.append(fileHandle("class2.txt"))
mainList.append(fileHandle("class3.txt"))

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

covMatrix = np.zeros((nClass,numFeature,numFeature))
scatterMatrix = np.zeros((nClass,numFeature,numFeature))
for i in range(nClass):
	extData = np.copy(mainList[i])
	for j in range(numFeature):
		extData[:, j] = np.subtract(extData[:, j], meanVector[i, j, 0])
	covMatrix[i] = np.matmul(np.transpose(extData), extData)
	scatterMatrix[i]=covMatrix[i]
	covMatrix[i] /= (numSample[i]-1)
	


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


testList = []
testList.append(fileHandle("classt1.txt"))
testList.append(fileHandle("classt2.txt"))
testList.append(fileHandle("classt3.txt"))
# projectedTestData=[]
# countlda=0

projectedDataMean=[]
countlda=0
for i in range(nClass):
	for j in range(i+1,nClass):
		# projectedTestData1=[]
		lst=[]
		lst.append(np.mean(projectedData[countlda][0]).reshape(1,1))
		lst.append(np.mean(projectedData[countlda][1]).reshape(1,1))
		projectedDataMean.append(lst)
		countlda+=1

projectedDataVar=[]
countlda=0
for i in range(nClass):
	for j in range(i+1,nClass):
		# projectedTestData1=[]
		lst=[]
		lst.append(np.var(projectedData[countlda][0]).reshape(1,1))
		lst.append(np.var(projectedData[countlda][1]).reshape(1,1))
		projectedDataVar.append(lst)
		countlda+=1


confusionMatrix=np.zeros((nClass,nClass),int)
for i in range(nClass):
	for j in range(len(testList[i])):
		countlda=0
		lst=np.zeros(nClass)
		for k in range(nClass):
			for l in range(k+1,nClass):
				proDataPoint=np.matmul(testList[i][j],directions[countlda]).reshape(1,1)
				p=discriminant(proDataPoint,projectedDataMean[countlda][0],projectedDataVar[countlda][0])
				q=discriminant(proDataPoint,projectedDataMean[countlda][1],projectedDataVar[countlda][1])
				if(p>q):
					lst[k]+=1
				else:
					lst[l]+=1
				countlda+=1
		confusionMatrix[i][np.argmax(lst)]+=1


# plot()
print(confusionMatrix)
#defining discriminant function