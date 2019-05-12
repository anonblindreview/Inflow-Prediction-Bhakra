import numpy as np
from keras.models import load_model
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

######### FIX RANDOM SEED FOR REPRODUCIBILITY ##
np.random.seed(7)

######### LOAD THE DATASET #####################
dataframe = read_csv('../data1.csv', usecols=[2], engine='python', skipfooter=3)

dataframe1 = read_csv('../data observed.csv', usecols=[5],engine='python', skipfooter=3)
dataframe.append(dataframe1,ignore_index = True)
dataset = dataframe.values
dataset = dataset.astype('float32')

######### NORMALIZE THE DATASET ################
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

######### LOAD THE MODEL #######################

path=input("what is type of model you wish to load?")
model=load_model("../weights/"+path+".h5")
print(model.summary())
batch_size=int(input("Enter batch size"))
look_back=3
number_of_predictions=len(dataframe1)-look_back-1
entries=dataset[-len(dataframe1):]
buffer=batch_size-(number_of_predictions)%batch_size
print("buffer length",buffer)
print("ghusne waale ki shape without buffer",entries.shape)
if buffer>0:
    ex=dataset[-buffer:]
    entries=np.append(ex,entries)
print("ghusne waale ki shape",entries.shape)
trainX, trainY = create_dataset(entries, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
print(trainX.shape)
######### RUN THE MODEL ########################
trainPredict = model.predict(trainX)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
dataset[-len(trainPredict):]=trainPredict.reshape(dataset.shape)
print(scaler.inverse_transform(dataset)[-len(trainPredict):])

