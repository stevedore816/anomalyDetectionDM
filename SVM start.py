import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from scipy.cluster import hierarchy as hc
import numpy as np

db = pd.read_csv("Data/benign_traffic.csv")
anomaly = pd.read_csv("Data/gafgyt_attacks/scan.csv")
"""Plan currently is to preprocess the data to make each feature have a mean of 0
and a std. of 1 across so that we can use correlation as a distance measure to than add to a distance correlation matrix
which will than result in clustering based on generic correlation for the AD ensemble algorithm"""

"""Alg to Find mean and std deviation"""
StdAvg = []

for column in db.keys():
    # This one line calculates the inverse of the mean and subtracts all the attributes by it so mean = 0, and than / by the standard deviation across making the std = 1
    db[column] = (db[column] + (-1 * db[column].mean())) / (db[column].std())
    print(db[column].mean(), db[column].std())
# Create Correlation Matrix
dbCorr = db.corr()
print(dbCorr)
for column in anomaly.keys():
    # This one line calculates the inverse of the mean and subtracts all the attributes by it so mean = 0, and than / by the standard deviation across making the std = 1
    anomaly[column] = (anomaly[column] + (-1 * anomaly[column].mean())) / (anomaly[column].std())


"""Logic for Autoencoder Implementation
Conditionals: Cannot exceed size 16 (no more than 7 autoencoders or 16 attributes per)
              Highly Correlated variable cannot exist inside of the same auto encoder
              Correlation >= .75 to be considered viable for entry into the respective autoencoder (including correlation to eachother pt 2)
              When the attribute is finished compiling do not compile that row & column again (avoids dupes)
                        **In later implementation, it may be a good idea to get rid of any attributes entering if this creates more than 7 autoencoders
Goals: Create multiple autoencoders that represent most (80%) of the attributes 
        These autoencoders will be based off of the highly correlated variables to a specific attribute
        Will create until either the size limit is filled or the timer stops
Implementation: The autoencoder is a list of list of attributes to be added to the queue 
"""
Key = dbCorr.keys()
autoencoders = []
explored = []
def createAutoEncoderAttribute(row):

    if len(autoencoders) < 9:
        autoencoder = []
        greater = True
        explored.append(row)
        for column in Key:
            if len(autoencoder) <= 15:
                if dbCorr[row][column] >= .75 and column not in explored:
                    autoencoder.append(column)

        smallest = 1000
        for encode in autoencoders:
            if len(encode) < smallest:
                smallest = len(encode)

        if len(autoencoder) != 0:
            avg = 0
            for attr in autoencoder:
                avg = avg + dbCorr[row][attr]
            avg = avg / len(autoencoder)

            for encoder in autoencoders:
                if avg < encoder[len(encoder) - 1]:
                    greater = False

        if greater and (len(autoencoder) > 5 or len(autoencoder) > smallest):
            autoencoder.append(avg)
            autoencoders.append(autoencoder)
            print(autoencoder,len(autoencoder), len(autoencoders))

for row in Key:
    createAutoEncoderAttribute(row)

def getAutoEncoder(num,anon):
    tempDB = pd.DataFrame()
    for i in range (0, len(autoencoders[num]) - 1):
        attr = autoencoders[num][i]
        if anon:
            tempDB[attr] = anomaly[attr]
        else:
            tempDB[attr] = db[attr]
    return tempDB
#autoencoder1 = getAutoEncoder(0,anon= False)
#anomaly1 = getAutoEncoder(0,anon = True)
#print(autoencoder1)


#Old method That I will delete alter doesn't do anything
def returnDB(search):
    Array = []
    for i in db.keys():
        # print(i[0:2])
        if i[0:2] == search:
            Array.append(i)
    # print(MI)
    DBMI = pd.DataFrame()

    for x in Array:
        DBMI[x] = db[x]
    return DBMI

"""This function will create a neural net with the previously provided information and create a autoencoder with it
autoencoderDB is the dataframe from pandas that handles the autoencoderDB
input is the shape that you are inputting into the neural network first
layers is the neurons in each layer(7 has to be odd) of the neural network
middle is the compression layer of the neural network or the point you want it at
disttest is a variable to see if we're testing the neural net based on dist or not
anomalyDB is the anomalydetection we're going to be using that correlate to those respective attr during testing
"""
matrixout = []
def createNeuralNet(autoencoderDB,input, layers, middle,disttest,anomalyDB):
    #autoencoderDB = returnDB('MI')
    train = autoencoderDB.iloc[:int(.8 * len(autoencoderDB)), :]
    trainx = train.iloc[int(.4 * len(train)):, :]
    validx = train.iloc[:int(.4 * len(train)), :]
    traintest = autoencoderDB.iloc[int(.8 * len(autoencoderDB)):, :]

    traintf = tf.convert_to_tensor(trainx)
    validtf = tf.convert_to_tensor(validx)
    tttf = tf.convert_to_tensor(traintest)
    #print(validtf, traintf)

    autoencoder = Sequential()
    autoencoder.add(Dense(layers[0], activation='elu', input_shape=(input,)))
    for i in range (1, len(layers)):
        if i == middle:
            autoencoder.add(Dense(layers[middle], activation='linear'))
        else:
            autoencoder.add(Dense(layers[i], activation='elu'))

    autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    trained = autoencoder.fit(traintf,traintf,epochs=15,validation_data=(validtf,validtf))

    print("Test Eval", autoencoder.evaluate(tttf,tttf))
    if disttest:
        test = autoencoderDB.iloc[13:14, :]
        testtf = tf.convert_to_tensor(test)
        target_data = autoencoder.predict(testtf)
        distg = np.linalg.norm(test - target_data, axis = -1)
        #print(dist)
        anon = anomalyDB.iloc[3:4,:]
        #print(anon)
        anon = tf.convert_to_tensor(anon)
        target_data = autoencoder.predict(anon)
        autoencoder.evaluate(anon,anon)
        dist = np.linalg.norm(anon - target_data, axis = -1)
        #print(dist)
        matrixout.append((distg,dist))
    return autoencoder
auto1 = createNeuralNet(getAutoEncoder(0,anon=False), 16, [10,8,4,2,4,8,10,16],3,disttest= True, anomalyDB=getAutoEncoder(0,anon=True))
auto2 = createNeuralNet(getAutoEncoder(1,anon=False), 16, [10,8,4,2,4,8,10,16],3,disttest= True, anomalyDB=getAutoEncoder(1,anon=True))
auto3 = createNeuralNet(getAutoEncoder(2,anon=False), 9, [8,4,2,4,8,9],2,disttest= True, anomalyDB=getAutoEncoder(2,anon=True))
auto4 = createNeuralNet(getAutoEncoder(3,anon=False), 6, [4,2,4,6],1,disttest= True, anomalyDB=getAutoEncoder(3,anon=True))
auto5 = createNeuralNet(getAutoEncoder(4,anon=False), 9, [8,4,2,4,8,9],2,disttest= True, anomalyDB=getAutoEncoder(4,anon=True))
auto6 = createNeuralNet(getAutoEncoder(5,anon=False), 7, [4,2,4,7],1,disttest= True, anomalyDB=getAutoEncoder(5,anon=True))
auto7 = createNeuralNet(getAutoEncoder(6,anon=False), 6, [4,2,4,6],1,disttest= True, anomalyDB=getAutoEncoder(6,anon=True))
print(matrixout)