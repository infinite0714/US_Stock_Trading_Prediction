import pandas as pd
import datetime
from datetime import date
import pandas_datareader.data as web
from matplotlib import pyplot as plt
import keras
from keras.initializers import HeUniform
from keras.layers import Input, Dense
from keras.losses import Huber
from keras.optimizers import Adam
from keras.models import Sequential, Model
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from pickle import dump, load
import numpy as np
import random
from collections import deque
from IPython.display import clear_output
import os

stocks = pd.read_csv('./dataset/data.csv')

num_days_per_company = 30

[idx, companynum] = stocks.shape
companynum = companynum - 1

input_shape = num_days_per_company*companynum
print(input_shape)
output_shape = companynum
print(output_shape)

def make_model(layers=[100, 100, 100], # number of hidden layers and number of neurons for hidden layers
               input_shape=input_shape, # input shape of model
               output_shape=output_shape, # output shape of model
               lr=0.001, # learning rate
               kernel_initializer=HeUniform, # kernel initializer
               hidden_activation='relu', # activation function for hidden layers
               output_activation='linear', # activation function for output layer
               loss=Huber,
               optimizer=Adam,
               metrics='accuracy'
              ):
    if not isinstance(metrics, list):
        metrics = [metrics]
    input_layer = Input(shape=input_shape)
    model = input_layer
    for layer in layers:
        model = Dense(units=layer, activation=hidden_activation, kernel_initializer=kernel_initializer())(model)
    model = Dense(output_shape, activation=output_activation, kernel_initializer=kernel_initializer())(model)
    model = Model(input_layer, model)
    model.compile(loss=loss(), optimizer=optimizer(lr=lr), metrics=metrics)
    return model

make_model().summary()