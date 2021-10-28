import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense,Input
from keras import optimizers
import numpy as np


db = pd.read_csv("Data/benign_traffic.csv")
anomaly = pd.read_csv("Data/gafgyt_attacks/scan.csv")

def returnDB(search):
    Array = []
    for i in db.keys():
        #print(i[0:2])
        if i[0:2] == search:
            Array.append(i)
    #print(MI)
    DBMI = pd.DataFrame()

    for x in Array:
        DBMI[x] = db[x]
    return DBMI

MI = returnDB("MI")
Hp = returnDB("Hp")
HH = returnDB("HH")
H = returnDB('H_')

train = MI.iloc[:int(.8*len(MI)),:]
trainx = MI.iloc[int(.4*len(train)):,:]
validx = MI.iloc[:int(.4*len(train)),:]
traintest = MI.iloc[int(.8*len(MI)):,:]

traintf = tf.convert_to_tensor(trainx)
validtf = tf.convert_to_tensor(validx)
tttf = tf.convert_to_tensor(traintest)
print(traintf)

autoencoder1 = Sequential()
autoencoder1.add(Dense(8, activation='elu',input_shape=(15,)))
autoencoder1.add(Dense(4,activation='elu'))
autoencoder1.add(Dense(2,activation='linear'))
autoencoder1.add(Dense(4,activation='elu')) #This is the compressed info
autoencoder1.add(Dense(8,activation='elu'))
autoencoder1.add(Dense(15, activation='linear'))
autoencoder1.compile(loss='mean_squared_error', optimizer='adam')
trained=autoencoder1.fit(traintf,traintf,epochs=60,validation_data=(validtf,validtf))

print(autoencoder1.evaluate(tttf,tttf))

test = MI.iloc[78:79, :]
testtf = tf.convert_to_tensor(test)
target_data = autoencoder1.predict(testtf)
dist = np.linalg.norm(test - target_data, axis = -1)
print(dist)
anon = anomaly.iloc[:1, :15]
print(anon)
anon = tf.convert_to_tensor(anon)
target_data = autoencoder1.predict(anon)
autoencoder1.evaluate(anon,anon)
dist = np.linalg.norm(anon - target_data, axis = -1)
print(dist)
