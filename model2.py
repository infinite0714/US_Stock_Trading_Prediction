import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas_datareader.data as web

# Define the Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define the environment for stock selection
class StockEnvironment:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = self._load_data()
        self.state_size = 5  # Number of features used as input state
        self.action_size = 2  # Number of actions (buy or sell)

    def _load_data(self):
        start_date = '2010-01-01'
        end_date = '2021-01-01'
        df = web.DataReader(self.symbol, 'yahoo', start_date, end_date)
        return df

    def get_state(self, t):
        state = []
        for i in range(t-self.state_size+1, t+1):
            state.append(self.data.iloc[i]['Close'])
        return np.array(state)

    def get_reward(self, action, t):
        if action == 0:  # Buy
            return self.data.iloc[t+1]['Close'] - self.data.iloc[t]['Close']
        elif action == 1:  # Sell
            return self.data.iloc[t]['Close'] - self.data.iloc[t+1]['Close']
        else:
            return 0

    def is_done(self, t):
        return t >= len(self.data) - self.state_size - 1

# Hyperparameters
symbol = 'AAPL'  # Example stock symbol (Apple Inc.)
num_episodes = 1000
batch_size = 32

# Create the environment and agent
env = StockEnvironment(symbol)
agent = DQNAgent(env.state_size, env.action_size)

# Training loop
for episode in range(num_episodes):
    state = env.get_state(0)
    state = np.reshape(state, [1, env.state_size])
    total_reward = 0
    for t in range(len(env.data) - env.state_size - 1):
        action = agent.act(state)
        reward = env.get_reward(action, t)
        next_state = env.get_state(t + 1)
        next_state = np.reshape(next_state, [1, env.state_size])
        done = env.is_done(t)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            print("Episode: {}/{}, Total Reward: {}".format(episode + 1, num_episodes, total_reward))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)