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

# Create Correlation Matrix
dbCorr = db.corr()
print(dbCorr)
for column in anomaly.keys():
    # This one line calculates the inverse of the mean and subtracts all the attributes by it so mean = 0, and than / by the standard deviation across making the std = 1
    anomaly[column] = (anomaly[column] + (-1 * anomaly[column].mean())) / (anomaly[column].std())
    print(anomaly[column].mean(), anomaly[column].std())



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
autoencoder1 = getAutoEncoder(0,anon= False)
#anomaly1 = getAutoEncoder(0,anon = True)
print(len(db))

def getAutoEncoderDB(num,db):
    temp = pd.DataFrame()
    for i in range (0, len(autoencoders[num]) - 1):
        attr = autoencoders[num][i]
        temp[attr] = [db[attr]]
    return temp

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
avgloss = []
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
    for i in range (1, len(layers) - 1):
        if i == middle:
            autoencoder.add(Dense(layers[middle], activation='linear'))
        else:
            autoencoder.add(Dense(layers[i], activation='elu'))
    autoencoder.add(Dense(layers[len(layers) - 1], activation='linear'))

    autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    trained = autoencoder.fit(traintf,traintf,epochs=10,validation_data=(validtf,validtf))
    autoencoder.evaluate(tttf,tttf)
    predictions = autoencoder.predict(traintf)
    dist = np.sqrt((traintf - predictions) * (traintf - predictions))
    #print(dist)
    std = float(tf.math.reduce_std(dist))
    mean = float(tf.reduce_mean(dist))
    avgloss.append((mean,std))
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
auto1 = createNeuralNet(getAutoEncoder(0,anon=False), 16, [10,8,4,1,4,8,10,16],3,disttest= False, anomalyDB=getAutoEncoder(0,anon=True))
auto2 = createNeuralNet(getAutoEncoder(1,anon=False), 16, [10,8,4,1,4,8,10,16],3,disttest= False, anomalyDB=getAutoEncoder(1,anon=True))
auto3 = createNeuralNet(getAutoEncoder(2,anon=False), 9, [8,4,1,4,8,9],2,disttest= False, anomalyDB=getAutoEncoder(2,anon=True))
auto4 = createNeuralNet(getAutoEncoder(3,anon=False), 6, [4,1,4,6],1,disttest= False, anomalyDB=getAutoEncoder(3,anon=True))
auto5 = createNeuralNet(getAutoEncoder(4,anon=False), 9, [8,4,1,4,8,9],2,disttest= False, anomalyDB=getAutoEncoder(4,anon=True))
auto6 = createNeuralNet(getAutoEncoder(5,anon=False), 7, [4,1,4,7],1,disttest= False, anomalyDB=getAutoEncoder(5,anon=True))
auto7 = createNeuralNet(getAutoEncoder(6,anon=False), 6, [4,1,4,6],1,disttest= False, anomalyDB=getAutoEncoder(6,anon=True))
print(avgloss)

"""Goal of this method is to return the distance from the predicted value vs the actual value"""
def predictOld(autoencoder,anomaly,index):
    flag = False
    factor = 2.5
    mean = avgloss[index][0]
    std = avgloss[index][1]
    anomaly = tf.convert_to_tensor(anomaly)
    target = autoencoder.predict(anomaly)
    dist = np.linalg.norm(anomaly - target)
   # dist = np.sqrt((anomaly - target) * (anomaly - target))
   # print(dist)
    if dist >= mean + (std * factor) or dist <= mean - (std * factor) :
        flag = True
    return flag

def EnsemblePrediction(subsection):
    votes = []
    test = getAutoEncoderDB(num=0, db=subsection)
    votes.append(predict(auto1,test,0))
    test = getAutoEncoderDB(num=1, db=subsection)
    votes.append(predict(auto2, test, 1))
    test = getAutoEncoderDB(num=2, db=subsection)
    votes.append(predict(auto3, test, 2))
    test = getAutoEncoderDB(num=3, db=subsection)
    votes.append(predict(auto4, test, 3))
    test = getAutoEncoderDB(num=4, db=subsection)
    votes.append(predict(auto5, test, 4))
    test = getAutoEncoderDB(num=5, db=subsection)
    votes.append(predict(auto6, test, 5))
    test = getAutoEncoderDB(num=6, db=subsection)
    votes.append(predict(auto7, test, 6))
    tot = 0
    print(votes)
    for vote in votes:
        if vote:
            tot = tot + 1
    if tot >= 4:
        return True
    return False



"""
Plan C
auto1.predict(tensoranon)
auto2.predict(tensoranon)
auto3.predict(tensoranon)
auto4.predict(tensoranon)
auto5.predict(tensoranon)
Plan Z
correct = 0
for i in range(0,len(anomaly)):
    subsection = anomaly.iloc[i,:]
    #print(subsection)
    if EnsemblePrediction(subsection):
        correct = correct + 1
avg1 = correct / len(anomaly)

correct = 0
for i in range(0,len(db)):
    subsection = db.iloc[i,:]
    #print(subsection)
    if EnsemblePrediction(subsection) == False:
        correct = correct + 1
avg2 = correct /len(db)
print(avg1, avg2)
"""
def predict(tensor,autoencoder,index):
    data = []
    factor = 3.4
    mean = avgloss[index][0]
    std = avgloss[index][1]
    prediction = autoencoder.predict(tensor)
    distance_matrix = (tensor - prediction)
    for distHD in distance_matrix:
        flag = False
        dist = np.linalg.norm(distHD)
        if dist >= mean + (std * factor) or dist <= mean - (std * factor):
            flag = True
        data.append(flag)
    return data
tensoranon1 = tf.convert_to_tensor(getAutoEncoder(0,anon=True))
output1 = predict(tensoranon1,auto1,0)
tensoranon2 = tf.convert_to_tensor(getAutoEncoder(1,anon=True))
output2 = predict(tensoranon2,auto2,1)
tensoranon3 = tf.convert_to_tensor(getAutoEncoder(2,anon=True))
output3 = predict(tensoranon3,auto3,2)
tensoranon4 = tf.convert_to_tensor(getAutoEncoder(3,anon=True))
output4 = predict(tensoranon4,auto4,3)
tensoranon5 = tf.convert_to_tensor(getAutoEncoder(4,anon=True))
output5 = predict(tensoranon5,auto5,4)
tensoranon6 = tf.convert_to_tensor(getAutoEncoder(5,anon=True))
output6 = predict(tensoranon6,auto6,5)
tensoranon7 = tf.convert_to_tensor(getAutoEncoder(6,anon=True))
output7 = predict(tensoranon7,auto7,6)
correct = 0
for i in range(0, len(anomaly)):
    count = 0
    if output1[i]: count = count + 1
    if output2[i]: count = count + 1
    if output3[i]: count = count + 1
    if output4[i]: count = count + 1
    if output5[i]: count = count + 1
    if output6[i]: count = count + 1
    if output7[i]: count = count + 1
    if count >= 4:
        correct = correct + 1
anoncorrect = correct / len(anomaly)
#print(correct / len(anomaly))

tensoranon1 = tf.convert_to_tensor(getAutoEncoder(0,anon=False))
output1 = predict(tensoranon1,auto1,0)
tensoranon2 = tf.convert_to_tensor(getAutoEncoder(1,anon=False))
output2 = predict(tensoranon2,auto2,1)
tensoranon3 = tf.convert_to_tensor(getAutoEncoder(2,anon=False))
output3 = predict(tensoranon3,auto3,2)
tensoranon4 = tf.convert_to_tensor(getAutoEncoder(3,anon=False))
output4 = predict(tensoranon4,auto4,3)
tensoranon5 = tf.convert_to_tensor(getAutoEncoder(4,anon=False))
output5 = predict(tensoranon5,auto5,4)
tensoranon6 = tf.convert_to_tensor(getAutoEncoder(5,anon=False))
output6 = predict(tensoranon6,auto6,5)
tensoranon7 = tf.convert_to_tensor(getAutoEncoder(6,anon=False))
output7 = predict(tensoranon7,auto7,6)
correct = 0
for i in range(0, len(db)):
    count = 0
    if output1[i]: count = count + 1
    if output2[i]: count = count + 1
    if output3[i]: count = count + 1
    if output4[i]: count = count + 1
    if output5[i]: count = count + 1
    if output6[i]: count = count + 1
    if output7[i]: count = count + 1
    if count < 4:
        correct = correct + 1
benigncorrect = correct / len(db)
print("Anomaly Score: ", anoncorrect, "Benign Score: ", benigncorrect)