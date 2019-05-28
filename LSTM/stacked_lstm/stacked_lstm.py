# Stacked LSTM for international airline passengers problem with memory
import numpy,sys
#import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
class ResetStateCallback(keras.callbacks.Callback):
        def on_epoch_begin(self,batch,logs={}):
                self.model.reset_states()
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
total_epochs=int(sys.argv[1])
dataframe = read_csv('../data1.csv',usecols=[2], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
print("trainX.shape",trainX.shape,"trainY.shape",trainY.shape)
print("testX.shape",testX.shape,"testY.shape",testY.shape)
#create a tensorboard object for logging purpose 
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
reset=ResetStateCallback()
#for i in range(100):
model.fit(trainX, trainY, epochs=total_epochs, batch_size=batch_size,validation_data=(testX, testY),callbacks=[tensorboard,reset], verbose=2, shuffle=False)
model.reset_states()
#save weigths
model.save("../weights/stacked_lstm.h5")
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)
print("trainPredict.shape",trainPredict.shape)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
print("testPredict.shape",testPredict.shape)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# save arrays / results to outline.npz for plotting
numpy.savez('outfile.npz', testPredictPlot=testPredictPlot , trainPredictPlot=trainPredictPlot, dataset=scaler.inverse_transform(dataset))
