import numpy as np 

nClass=3
learningRate=0.01
nIteration=1000

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

def updateWeights(weight,firstClassData,secondClassData):
	for i in range(nIteration):
		nMissClassify=0
		avgErrorData=np.zeros((1,nFeature+1),float)
		h=np.dot(firstClassData,weight)
		for j in range(len(h)):
			if(h[j]<0):
				avgErrorData+=firstClassData[j]
				nMissClassify+=1
		h=np.dot(secondClassData,weight)
		for j in range(len(h)):
			if(h[j]>0):
				avgErrorData+=-secondClassData[j]
				nMissClassify+=1
		if(nMissClassify!=0):
			avgErrorData/=nMissClassify
			weight=(weight+learningRate*np.transpose(avgErrorData)).reshape(nFeature+1,1)
	return weight





mainList = []
mainList.append(fileHandle("class1.txt"))
mainList.append(fileHandle("class2.txt"))
mainList.append(fileHandle("class3.txt"))

testList = []
testList.append(fileHandle("classt1.txt"))
testList.append(fileHandle("classt2.txt"))
testList.append(fileHandle("classt3.txt"))

nFeature=len(mainList[0][0])

weights=[]
for i in range(nClass):
	for j in range(i+1,nClass):
		weight=np.random.random((nFeature+1,1))
		# weight=np.zeros((nFeature+1,1),float)
		ipData1=np.insert(mainList[i],0,1,axis=1)
		ipData2=np.insert(mainList[j],0,1,axis=1)
		weights.append(updateWeights(weight,ipData1,ipData2))


confusionMatrix=np.zeros((nClass,nClass),int)
for tst in range(nClass):
	for tastex in range(len(testList[tst])):
		voting=np.zeros(nClass,int)
		testPoint=np.insert(testList[tst][tastex],0,1).reshape(nFeature+1,1)
		count1=0
		for i in range(nClass):
			for j in range(i+1,nClass):
				p=np.dot(np.transpose(weights[count1]),testPoint)
				count1+=1
				if(p>0):
					voting[i]+=1
				else:
					voting[j]+=1
		confusionMatrix[tst][np.argmax(voting)]+=1

print(confusionMatrix)

				

