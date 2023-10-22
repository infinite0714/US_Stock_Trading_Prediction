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
from sklearn.impute import SimpleImputer

class Company:
  def __init__(self, symbol, name, stock):
    self.symbol = symbol
    self.name = name
    self.stock = stock

symbols = []
names = []

today = datetime.datetime.now()
print("Today is " + str(today.strftime("%b-%d-%Y")))
fifteen_years_ago = today - datetime.timedelta(days=15*365)
print("The day fifteen years ago was " + str(fifteen_years_ago.strftime("%b-%d-%Y")))


table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
for i in range(len(df)):
    if df.loc[i, "Symbol"] == "BF.B": # finance can't find data for BF.B, symbol may be delisted
        continue
    date_first_added = df.loc[i, "Date added"]
    if not isinstance(date_first_added, str):
        continue
    date_first_added_len = len(date_first_added)
    if date_first_added_len < 7 and date_first_added_len > 1:
        if date_first_added[date_first_added_len - 1] == "?":
            date_first_added = date_first_added[:date_first_added_len - 1] + "-01-01"
        else:
            date_first_added = date_first_added + "-01-01"
    elif date_first_added_len >= 7 and date_first_added_len < 10:
        if date_first_added[date_first_added_len - 1] == "?":
            date_first_added = date_first_added[:date_first_added_len - 1] + "-01"
        else:
            date_first_added = date_first_added + "-01"
    date_first_added = date_first_added[:10]
    date_first_added = datetime.datetime.strptime(date_first_added, "%Y-%m-%d")
    if date_first_added > fifteen_years_ago:
        continue
    symbols.append(df.loc[i, "Symbol"])
    names.append(df.loc[i, "Security"])

print(len(symbols))
print(len(names))

# ticker_dict = {}
# for idx, ticker in enumerate(symbols):
#     try:
#         df_ticker = web.DataReader(ticker, 'iex', start=fifteen_years_ago, end=today, api_key='pk_e11fa07e62fb48cbbe64aff1e0da7570') 
#         ticker_dict[ticker] = df_ticker['close']
#     except:pass
# stocks = pd.DataFrame(ticker_dict) 

# print(stocks.head())

# stocks.to_csv('./dataset/data.csv')

stocks = pd.read_csv('./dataset/data.csv')
# Analyze Dataset
companies = []
for i in range(len(symbols)):
    symbol = symbols[i]
    name = names[i]
    companies.append(Company(symbol, name, list(stocks[symbol][0:])))

# def plot_stock(symbol):
#   for i in range(len(companies)):
#     if companies[i].symbol == symbol:
#       plt.plot(companies[i].stock)
#       plt.title(companies[i].name + " Stock")
#       plt.ylabel("Stock Price (USD)")
#       plt.xlabel(f"Number of days (after {fifteen_years_ago.month}/{fifteen_years_ago.day}/{fifteen_years_ago.year})")
#       plt.show()
#       return
#   print(symbol + " stock data not found")

# plot_stock("GOOG")

num_days_per_company = 30

x = []
y = []

for i in range(num_days_per_company, len(companies[0].stock)):
    x_i = []
    y_i = []
    for j in range(len(companies)):
        x_i.extend(companies[j].stock[i-num_days_per_company:i])
        if np.isnan(companies[j].stock[i]) == True:
            y_i.append(0)
            continue
        y_i.append(companies[j].stock[i])
    x.append(x_i)
    y.append(y_i)

x = np.array(x)
y = np.array(y)

imputer = SimpleImputer(strategy="mean")
imputer.fit(x)
x = imputer.transform(x)

print(x.shape)
print(y.shape)

np.save('./dataset/x.npy', x)
np.save('./dataset/y.npy', y)

train_portion = 2.0/3.0
val_portion = 1.0/6.0
test_portion = 1.0/6.0

train_idx = int(x.shape[0]*train_portion)
val_idx = int(x.shape[0]*val_portion)+train_idx
test_idx = x.shape[0]+1

x_train = x[:train_idx]
x_val = x[train_idx:val_idx]
x_test = x[val_idx:test_idx]

y_train = y[:train_idx]
y_val = y[train_idx:val_idx]
y_test = y[val_idx:test_idx]

print(x_train.shape)
print(y_train.shape)
print('/n')
print(x_val.shape)
print(y_val.shape)
print('/n')
print(x_test.shape)
print(y_test.shape)

normalizer_x = Normalizer()
normalizer_y = Normalizer()

x_train = normalizer_x.fit_transform(x_train)
y_train = normalizer_y.fit_transform(y_train)

x_val = normalizer_x.transform(x_val)
y_val = normalizer_y.transform(y_val)

x_test = normalizer_x.transform(x_test)
y_test = normalizer_y.transform(y_test)

dump(normalizer_x, open('./preprocessed_data/normalizer_x.pkl', 'wb'))
dump(normalizer_y, open('./preprocessed_data/normalizer_y.pkl', 'wb'))

np.save("./preprocessed_data/x_train.npy", x_train)
np.save("./preprocessed_data/y_train.npy", y_train)

np.save("./preprocessed_data/x_val.npy", x_val)
np.save("./preprocessed_data/y_val.npy", y_val)

np.save("./preprocessed_data/x_test.npy", x_test)
np.save("./preprocessed_data/y_test.npy", y_test)