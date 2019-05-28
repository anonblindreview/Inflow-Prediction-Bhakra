import numpy as np
from keras.models import load_model
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
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

######### MODEL DATA FEEDING ###################
test = dataset[-len(dataframe1):,:]
batch_size=int(input("Enter batch size"))
look_back=3
number_of_predictions=len(dataframe1)-look_back
print(f"batch size: {batch_size} number of predictions: {number_of_predictions}")
buffer=batch_size-(number_of_predictions)%batch_size
print(f"buffer: {buffer}")
if buffer>0:
    ex=dataset[-buffer:]
    test=np.append(ex,test)
testX, testY = create_dataset(test, look_back)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
######### RUN THE MODEL ########################
testPredict = model.predict(testX, batch_size=batch_size)
testPredict = scaler.inverse_transform(testPredict)
for index in range(number_of_predictions):
    print(f"input: {testX[index]} prediction: {testPredict[index]}")

