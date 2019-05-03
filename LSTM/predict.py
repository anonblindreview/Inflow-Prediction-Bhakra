import numpy
import matplotlib.pyplot as plt
import math
from tensorflow.python import keras
from keras.models import load_model
#loading model
def create_dataset(dataset, look_back=3):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
path=input("what is type of model you wish to load?")
model.load("weigths/"+path+".h5")
print(model.summary())
batch_size=1
dataset=[1,2,3,4]
trainX=create_dataset(dataset)
months=[31,28,31,30,31,30,31,31,30,31,30,31]
for i in range(sum(months[:4])):
    trainPredict = model.predict(trainX, batch_size=batch_size)
    print(trainPredict)
    dataset=dataset[:-1]
    dataset.append(trainPredict)
    trainX=create_dataset(dataset)
    
