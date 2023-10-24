import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from alpha_vantage.timeseries import TimeSeries

# Alpha Vantage API credentials
API_KEY = 'P4A6X4Y0GTXJBJBU'

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

# Define the environment for stock trading
class StockEnvironment:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = self._load_data()
        self.state_size = 5  # Number of features used as input state
        self.action_size = 3  # Number of actions (buy, sell, hold)

    def _load_data(self):
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=self.symbol, outputsize='full')
        return data

    def get_state(self, t):
        state = []
        for i in range(t - self.state_size + 1, t + 1):
            state.append(self.data.iloc[i]['4. close'])
        return np.array(state)

    def get_reward(self, action, t):
        if action == 0:  # Buy
            return self.data.iloc[t + 1]['4. close'] - self.data.iloc[t]['4. close']
        elif action == 1:  # Sell
            return self.data.iloc[t]['4. close'] - self.data.iloc[t + 1]['4. close']
        else:  # Hold
            return 0

    def is_done(self, t):
        return t >= len(self.data) - self.state_size - 1

# Hyperparameters
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Example stock symbols (Apple, Google, Microsoft, Amazon)
num_episodes = 1000
batch_size = 32

# Create the environment and agent
envs = [StockEnvironment(symbol) for symbol in symbols]
agent = DQNAgent(envs[0].state_size, envs[0].action_size)

# Training loop
for episode in range(num_episodes):
    total_reward = 0
    for env in envs:
        state = env.get_state(0)
        state = np.reshape(state, [1, env.state_size])
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
                print("Stock: {}, Episode: {}/{}, Total Reward: {}".format(env.symbol, episode + 1, num_episodes, total_reward))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)



import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from alpha_vantage.timeseries import TimeSeries

# Alpha Vantage API credentials
API_KEY = 'your_api_key'

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

# Define the environment for stock trading
class StockEnvironment:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = self._load_data()
        self.state_size = 5  # Number of features used as input state
        self.action_size = 1  # Number of actions (predict stock value)

    def _load_data(self):
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=self.symbol, outputsize='full')
        return data

    def get_state(self, t):
        state = []
        for i in range(t - self.state_size + 1, t + 1):
            state.append(self.data.iloc[i]['4. close'])
        return np.array(state)

    def get_reward(self, action, t):
        return self.data.iloc[t + 1]['4. close']  # Reward is the next day's stock value

    def is_done(self, t):
        return t >= len(self.data) - self.state_size - 1

# Hyperparameters
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Example stock symbols (Apple, Google, Microsoft, Amazon)
num_episodes = 1000
batch_size = 32

# Create the environment and agent
envs = [StockEnvironment(symbol) for symbol in symbols]
agent = DQNAgent(envs[0].state_size, envs[0].action_size)

# Training loop
for episode in range(num_episodes):
    total_reward = 0
    for env in envs:
        state = env.get_state(0)
        state = np.reshape(state, [1, env.state_size])
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
                print("Stock: {}, Episode: {}/{}, Total Reward: {}".format(env.symbol, episode + 1, num_episodes, total_reward))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)