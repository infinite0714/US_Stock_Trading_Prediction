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

normalizer_x = load(open("./preprocessed_data/normalizer_x.pkl", "rb"))
normalizer_y = load(open("./preprocessed_data/normalizer_y.pkl", "rb"))

x = np.load("./dataset/x.npy")
y = np.load("./dataset/y.npy")

x_train = np.load("./preprocessed_data/x_train.npy")
y_train = np.load("./preprocessed_data/y_train.npy")

x_val = np.load("./preprocessed_data/x_val.npy")
y_val = np.load("./preprocessed_data/y_val.npy")

x_test = np.load("./preprocessed_data/x_test.npy")
y_test = np.load("./preprocessed_data/y_test.npy")

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

def get_q_values(model, stock):
    return model.predict(np.array([stock]))[0]

def train(replay_memory, model, target_model, end, lr=0.7, discount=0.618, min_replay_size=1000, batch_size=128):

    if len(replay_memory) < min_replay_size:
        return

    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, end) in enumerate(mini_batch):
        if not end:
            max_future_q = reward + discount * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - lr) * current_qs[action] + lr * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def tanh(x):
  return (np.exp(2*x)-1)/(np.exp(2*x)+1)

plt.plot([tanh(0.05*i) for i in range(-100, 101)])

def step(i, action, a=0.05):
  stock_price = x_train[i][num_days_per_company*action + num_days_per_company - 1]
  next_day_stock_price = y_train[i][action]
  diff = next_day_stock_price - stock_price
  percent_diff = diff/stock_price
  reward = tanh(a*percent_diff)
  done = False
  new_i = i+1
  if new_i >= x_train.shape[0]-1:
    done = True
  new_observation = x_train[new_i]
  return [new_i, new_observation, reward, done]

def fit_model(
    layers=[100, 100, 100], # hidden layers for model
    train_episodes = 30,
    epsilon=1, # initial epsilon value
    max_epsilon=1, # maximum epsilon value
    min_epsilon=0.01, # minimum epsilon value
    decay = 0.01,
    update=4, # updates model every update steps
    target_update=100 # updates target_model every target_update steps
):

  model = make_model(layers=layers)
  target_model = make_model(layers=layers)
  target_model.set_weights(model.get_weights())

  replay_memory = deque(maxlen=50000)

  target_update_counter = 0

  X = []
  y = []

  steps = 0

  for episode in range(train_episodes):
      total_training_rewards = 0
      i = 0
      observation = x_train[i]
      done = False
      while not done:
          progress_bar = list('**********')
          for k in range(int(10*i/x_train.shape[0])):
            progress_bar[k] = '-'
          progress_bar = "".join(progress_bar)
          print(f"Episode {episode+1}/{train_episodes}: {i}/{x_train.shape[0]} {progress_bar[:5]}{round(100*i/x_train.shape[0], 2)}%{progress_bar[5:]}")
          steps += 1

          random_number = np.random.rand()
          if random_number <= epsilon:
              action = np.random.randint(companynum)
          else:
              predicted = get_q_values(model, observation)
              action = np.argmax(predicted)
          new_i, new_observation, reward, done = step(i, action)
          replay_memory.append([observation, action, reward, new_observation, done])

          # 3. Update the Main Network using the Bellman Equation
          if steps % update == 0 or done:
              train(replay_memory, model, target_model, done)

          observation = new_observation
          i = new_i
          total_training_rewards += reward

          if done:
              print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
              total_training_rewards += 1

              if steps >= 100:
                  print('Copying main network weights to the target network weights')
                  target_model.set_weights(model.get_weights())
                  steps = 0
              break

      epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
  return model, target_model

models = []

for i in range(1, 5): # number of 100 neuron hidden layers for model
  model, target_model = fit_model(layers=[100 for k in range(i)])
  model.save(f"./models/model_{i}")
  models.append(model)
