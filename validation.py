import numpy as np
import keras
from pickle import load, dump

class Company:
  def __init__(self, symbol, name, stock):
    self.symbol = symbol
    self.name = name
    self.stock = stock

num_days_per_company = 30

normalizer_x = load(open("./preprocessed_data/normalizer_x.pkl", "rb"))
normalizer_y = load(open("./preprocessed_data/normalizer_y.pkl", "rb"))

x_train = np.load("./preprocessed_data/x_train.npy")
train_idx = x_train.shape[0]

x_val = np.load("./preprocessed_data/x_val.npy")
y_val = np.load("./preprocessed_data/y_val.npy")

x_test = np.load("./preprocessed_data/x_test.npy")
y_test = np.load("./preprocessed_data/y_test.npy")

model = keras.models.load_model('./models/model_1')

with open('./preprocessed_data/file.pkl', 'rb') as file: 
    companies = load(file)

def stock_growth(company, i, j):
  return companies[company].stock[j]/companies[company].stock[i]

def invest(model, stocks):
  return np.argmax(model.predict(np.array([stocks]))[0])

def get_balances(model, stocks_x, buffer_idx):
  balances = []
  balance = 1
  balances.append(balance)
  for i in range(stocks_x.shape[0]-1):
    balance *= stock_growth(invest(model, stocks_x[i]), i+buffer_idx+num_days_per_company, i+buffer_idx+num_days_per_company+1)
    balances.append(balance)
  return (balances, balance)

balances = []
final_balances = []

balance, final_balance = get_balances(model, x_val, train_idx)