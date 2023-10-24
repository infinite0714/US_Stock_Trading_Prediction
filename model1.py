import numpy as np
import pandas as pd
from googlefinance import getQuotes
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Define a function to preprocess the stock data
def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    return scaled_data

# Define a function to create the LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Get a list of all S&P 500 ticker symbols
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500_tickers = list(sp500_tickers['Symbol'])

# Define the time period for stock data
start_date = '2020-01-01'
end_date = '2021-01-01'

# Fetch stock data for each ticker symbol and preprocess it
scaled_data = []
for ticker in sp500_tickers:
    try:
        stock_info = getQuotes(ticker, start_date=start_date, end_date=end_date)
        stock_prices = [float(info['LastTradePrice']) for info in stock_info]
        scaled_data.append(preprocess_data(stock_prices))
    except:
        print(f"Error fetching data for {ticker}")

# Combine the scaled data into a single numpy array
combined_data = np.concatenate(scaled_data, axis=1)

# Split the data into training and testing sets
train_size = int(0.8 * len(combined_data))
train_data = combined_data[:train_size]
test_data = combined_data[train_size:]

# Prepare the training and testing data
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create and train the LSTM model
model = create_model(input_shape=(X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the testing data
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Make predictions on the testing data
predictions = model.predict(X_test)

# Print the predicted and actual stock prices
for i in range(len(predictions)):
    print(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")