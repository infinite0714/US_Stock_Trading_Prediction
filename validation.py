import numpy as np
import keras
from pickle import load, dump
import matplotlib.pyplot as plt
from model import *

class Company:
  def __init__(self, symbol, name, stock):
    self.symbol = symbol
    self.name = name
    self.stock = stock

num_days_per_company = 30

normalizer_x = load(open("./preprocessed_data/normalizer_x.pkl", "rb"))
normalizer_y = load(open("./preprocessed_data/normalizer_y.pkl", "rb"))

x_train = np.load("./preprocessed_data/x_train.npy")

x_val = np.load("./preprocessed_data/x_val.npy")
y_val = np.load("./preprocessed_data/y_val.npy")

x_test = np.load("./preprocessed_data/x_test.npy")
y_test = np.load("./preprocessed_data/y_test.npy")

model = keras.models.load_model('./models/model_1')
company_logs = []

train_idx = x_train.shape[0]
val_idx = x_val.shape[0] + train_idx

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
    # print(invest(model, stocks_x[i]))
    company_logs.append(invest(model, stocks_x[i]))
    balance *= stock_growth(invest(model, stocks_x[i]), i+buffer_idx+num_days_per_company, i+buffer_idx+num_days_per_company+1)
    balances.append(balance)
  return (balances, balance)


# balance, final_balance= get_balances(model, x_val, train_idx)

# plt.plot(balance)
# plt.title("Total balance of Stock Bot accounts")
# plt.ylabel("Balance (USD)")
# plt.xlabel("Days")
# plt.legend("model")
# plt.show()



balance_result, final_balance_result= get_balances(model, x_test, val_idx)

plt.plot(balance_result)
plt.title("Total balance of Stock Bot accounts")
plt.ylabel("Balance (USD)")
plt.xlabel("Days")
plt.legend("model")
plt.show()