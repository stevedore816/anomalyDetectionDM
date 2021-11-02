import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
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

    if len(autoencoders) < 8:
        autoencoder = []
        greater = True
        explored.append(row)
        for column in Key:
            if len(autoencoder) <= 15:
                if dbCorr[row][column] >= .75 and column not in explored:
                    autoencoder.append(column)
                    #explored.append(column)

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

"""
#Naive Approach
MI = returnDB("MI")
Hp = returnDB("Hp")
HH = returnDB("HH")
H = returnDB('H_')

train = MI.iloc[:int(.8 * len(MI)), :]
trainx = train.iloc[int(.4 * len(train)):, :]
validx = train.iloc[:int(.4 * len(train)), :]
traintest = MI.iloc[int(.8 * len(MI)):, :]

traintf = tf.convert_to_tensor(trainx)
validtf = tf.convert_to_tensor(validx)
tttf = tf.convert_to_tensor(traintest)
print(traintf)

#This is all neural net stuff but preprocessing of the values still needs to happen before I worry about this
autoencoder1 = Sequential()
autoencoder1.add(Dense(10, activation='elu',input_shape=(15,)))
autoencoder1.add(Dense(8, activation='elu'))
autoencoder1.add(Dense(4,activation='linear'))
autoencoder1.add(Dense(8, activation='elu'))
autoencoder1.add(Dense(10, activation='elu'))
autoencoder1.add(Dense(15, activation='linear'))

autoencoder1.compile(loss='mean_squared_error', optimizer='adam')
trained=autoencoder1.fit(traintf,traintf,epochs=15,validation_data=(validtf,validtf))

print(autoencoder1.evaluate(tttf,tttf))

test = MI.iloc[:1, :15]
testtf = tf.convert_to_tensor(test)s
target_data = autoencoder1.predict(testtf)
dist = np.linalg.norm(test - target_data, axis = -1)
print(dist)
anon = anomaly.iloc[1:2, :15]
print(anon)
anon = tf.convert_to_tensor(anon)
target_data = autoencoder1.predict(anon)
autoencoder1.evaluate(anon,anon)
dist = np.linalg.norm(anon - target_data, axis = -1)
print(dist)
"""