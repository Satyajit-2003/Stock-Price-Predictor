import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout

API_KEY = 'YOUR_API_KEY'

# Getting the dataset
def get_data(stock_key):
    url  = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={stock_key}&outputsize=full&apikey={API_KEY}'
    data = requests.get(url).json()

    file = open(f'{stock_key}.csv', 'w', newline='')
    csv_w = csv.writer(file)
    csv_w.writerow(['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient'])

    for i in data['Time Series (Daily)']:
        csv_w.writerow([i, data['Time Series (Daily)'][i]['1. open'], data['Time Series (Daily)'][i]['2. high'], data['Time Series (Daily)'][i]['3. low'], data['Time Series (Daily)'][i]['4. close'], data['Time Series (Daily)'][i]['5. adjusted close'], data['Time Series (Daily)'][i]['6. volume'], data['Time Series (Daily)'][i]['7. dividend amount'], data['Time Series (Daily)'][i]['8. split coefficient']])
    file.close()

    df = pd.read_csv(f'{stock_key}.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    train = df[int(0.8*(len(df))):]
    test = df[:int(0.8*(len(df)))]

    train.to_csv(f'{stock_key}_train.csv', index=False)
    test.to_csv(f'{stock_key}_test.csv', index=False)

def training_model(stock_key):
    # Getting the training dataset
    print("Getting the training dataset...")
    data = pd.read_csv(f'{stock_key}_train.csv')

    # Cleaning the dataset
    print("Cleaning the dataset...")
    data['date'] = pd.to_datetime(data['date'])
    data = data.dropna()

    # Getting the training data
    trainData = data.iloc[:, 5:6].values

    # Feature Scaling
    print("Feature Scaling...")
    sc = MinMaxScaler(feature_range=(0, 1))
    trainData = sc.fit_transform(trainData)

    # Convert the data into a numpy array
    print("Converting the data into a numpy array...")
    x_train = []
    y_train = []
    for i in range(60, len(trainData)):
        x_train.append(trainData[i-60:i, 0])
        y_train.append(trainData[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshaping the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)

    # Building the LSTM model
    print("Building the LSTM model...")
    model = Sequential()

    # Adding the first LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # Adding the second LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding the third LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding the fourth LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Compiling the model
    print("Compiling the model...")
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(model.summary())

    # Training the model
    print("Training the model...")
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=2)
    input()

    # Plotting the loss
    print("Plotting the loss...")
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # Saving the model
    print("Saving the model...")
    model.save(f'{stock_key}_model.h5')

def testing_model(stock_key):
    model = load_model(f'{stock_key}_model.h5')

    # Getting the testing dataset
    print("Getting the testing dataset...")
    data = pd.read_csv(f'{stock_key}_test.csv')

    # Cleaning the dataset
    print("Cleaning the dataset...")
    data['date'] = pd.to_datetime(data['date'])
    dates = data['date'].iloc[60:].values
    data = data.dropna()
    testData = data.iloc[:, 5:6]

    # Extract the target values
    y_test = testData.iloc[60:, 0:].values

    # Extract the input values and perform any necessary transformations
    input_close = testData.iloc[:, 0:].values

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    testData = sc.fit_transform(testData)

    # Convert the data into a numpy array
    input_close = sc.transform(input_close)
    print(input_close.shape)

    # Reshape the input data
    x_test = []
    length = len(testData)
    timestep = 60
    for i in range(timestep, length):
        x_test.append(input_close[i-timestep:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict the values
    print("Predicting the values...")
    predicted_close = model.predict(x_test)
    predicted_close = sc.inverse_transform(predicted_close)

    # Calculate the root mean squared error (RMSE)
    print("Calculating the root mean squared error (RMSE)...")
    rmse = np.sqrt(np.mean(((predicted_close - y_test)**2)))

    # Calculate the mean absolute percentage error (MAPE)
    print("Calculating the mean absolute percentage error (MAPE)...")
    mape = np.mean(np.abs((predicted_close - y_test)/y_test))

    # Plot the results
    print("Plotting the results...")
    plt.figure(figsize=(14,5))
    plt.plot(dates, y_test, color='red', label='Real Stock Price')
    plt.plot(dates, predicted_close, color='blue', label='Predicted Stock Price')

    # Format the x-axis as dates
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.gcf().autofmt_xdate()

    # Showing errors as text
    plt.text(0.01, 0.9, 'RMSE = %.4f' %(rmse), ha='left', va='center', transform=plt.gca().transAxes)
    plt.text(0.01, 0.85, 'MAPE = %.4f' %(mape), ha='left', va='center', transform=plt.gca().transAxes)

    # Set the title and labels
    plt.title(f'{stock_key} Stock Price Prediction')
    plt.ylabel('Stock Price')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    plt.show()

def main():
    stock_key = input("Enter the stock key: ")
    get_data(stock_key)
    training_model(stock_key)
    testing_model(stock_key)


if __name__ == "__main__":
    main()



