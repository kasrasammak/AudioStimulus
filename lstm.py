#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:50:48 2020

@author: owlthekasra
"""
# lstm model
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import tensorflow as tf
import methods as md
#Read the file to a pandas object
#data=pd.read_csv('filedir')
#convert the pandas object to a tensor

def fit_model(trainX, trainy, batch=64, epoch=15, activation_1='relu', activation_2='softmax'):
	verbose, epochs, batch_size = 0, epoch, batch
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation=activation_1))
	model.add(Dense(n_outputs, activation=activation_2))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)   
	return model, batch_size

def eval_model(model, testX,testy, batch_size):
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


def get_formatted_x_y(df):
    x = np.array([np.array(df[0].iloc[:, 1:].T), np.array(df[1].iloc[:, 1:].T), np.array(df[2].iloc[:, 1:].T), np.array(df[3].iloc[:, 1:].T)])
    x_t = x.T
    y = np.array([np.array(df[0].iloc[:, 0]), np.array(df[1].iloc[:, 0]), np.array(df[2].iloc[:, 0]), np.array(df[3].iloc[:, 0])])
    y_t = y.T
    return (x_t, y_t)
#tensX = tf.convert_to_tensor(yoyo)
#tensy = tf.convert_to_tensor(ytrainyo)
#
xtrain, ytrain = get_formatted_x_y(training_chans)
xtest, ytest = get_formatted_x_y(validation_chans)
model, batch_size = fit_model(xtrain, ytrain)
acc = eval_model(model, xtest, ytest, batch_size)




doit = training_chans[2].iloc[:, 1:].reset_index().iloc[:, 1:]
doittoit = training_chans[2].iloc[:, 0].reset_index().iloc[:, 1:]
toit = validation_chans[2].iloc[:, 1:].reset_index().iloc[:, 1:]
toitdoit = validation_chans[2].iloc[:, 0].reset_index().iloc[:, 1:]

model2, acc2, pred2, y_test2 = md.fitPredictValSet(training_chans[1].iloc[:, 1:], training_chans[1].iloc[:, 0], validation_chans[1].iloc[:, 1:], validation_chans[1].iloc[:, 0], 'knn')
model2, acc2, pred2, y_test2 = md.fitPredictValSet(doit, doittoit, toit, toitdoit, 'knn')

# bgRand = bg.sample(frac=1)

# big1 = bg.iloc[::4, :]
# big2 = bg.iloc[1::4, :]
# big3 = bg.iloc[2::4, :]
# big4 = bg.iloc[3::4, :]

# resetbig= big1.reset_index().index.values
# resetbig = np.random.shuffle(resetbig).tolist()

# bgRand1 = big1.iloc[resetbig]
# bgRand2 = big2.iloc[resetbig]
# bgRand3 = big3.iloc[resetbig]
# bgRand4 = big4.iloc[resetbig]

# bigX1 = bgRand1.iloc[:, 1:]
# bigX2 = bgRand2.iloc[:, 1:]
# bigX3 = bgRand3.iloc[:, 1:]
# bigX4 = bgRand4.iloc[:, 1:]

# bigy1 = bgRand1.iloc[:, 0]
# bigy2 = bgRand2.iloc[:, 0]
# bigy3 = bgRand3.iloc[:, 0]
# bigy4 = bgRand4.iloc[:, 0]

# def cut_by(df, num):
#     train = df.iloc[:num]
#     val = df.iloc[num:]
#     return train, val
# trainX1, valX1 = cut_by(bigX1, 650)
# trainX2, valX2 = cut_by(bigX2, 650)
# trainX3, valX3 = cut_by(bigX3, 650)
# trainX4, valX4 = cut_by(bigX4, 650)

# trainy1, valy1 = cut_by(bigy1, 650)
# trainy2, valy2 = cut_by(bigy2, 650)
# trainy3, valy3 = cut_by(bigy3, 650)
# trainy4, valy4 = cut_by(bigy4, 650)

# vyovyovyo = np.array([np.array(valX1.T), np.array(valX2.T), np.array(valX3.T), np.array(valX4.T)])
# vyovyo = vyovyovyo.T
# vytrainvyo = np.array([np.array(valy1.T), np.array(valy2.T), np.array(valy3.T), np.array(valy4.T)])
# vytranvyo = vytrainvyo.T






