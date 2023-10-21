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

def plot_stock(symbol):
  for i in range(len(companies)):
    if companies[i].symbol == symbol:
      plt.plot(companies[i].stock)
      plt.title(companies[i].name + " Stock")
      plt.ylabel("Stock Price (USD)")
      plt.xlabel(f"Number of days (after {fifteen_years_ago.month}/{fifteen_years_ago.day}/{fifteen_years_ago.year})")
      plt.show()
      return
  print(symbol + " stock data not found")

plot_stock("GOOG")