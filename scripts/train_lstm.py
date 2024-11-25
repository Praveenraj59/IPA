import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to calculate Moving Average
def moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate RSI (Relative Strength Index)
def rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to preprocess the data
def preprocess_data(data, time_step=60):
    # Ensure 'Close' column is numeric
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # Drop rows with NaN values
    data.dropna(subset=['Close'], inplace=True)

    # Add technical indicators
    data['MA50'] = moving_average(data, 50)
    data['RSI14'] = rsi(data)

    # Drop rows with NaN values from the indicators
    data.dropna(inplace=True)

    # Extract the 'Close' price for LSTM prediction
    close_data = data['Close'].values
    close_data = close_data.reshape(-1, 1)

    # Normalize the data (MinMax Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    # Prepare the data for LSTM (use past 60 days to predict the next day's price)
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])  # Predict 'Close' price for the next day

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Shape the data for LSTM (samples, time_steps, features)

    return X, y, scaler

# Load and prepare data for multiple stocks (e.g., AAPL, TSLA, GOOGL)
def load_and_preprocess_data():
    # Load data for multiple stocks
    aapl_data = pd.read_csv('../data/AAPL_processed_data.csv')
    tsla_data = pd.read_csv('../data/TSLA_processed.csv')
    googl_data = pd.read_csv('../data/GOOGL_processed.csv')

    # Preprocess data for each stock
    aapl_X, aapl_y, scaler = preprocess_data(aapl_data)
    tsla_X, tsla_y, scaler = preprocess_data(tsla_data)
    googl_X, googl_y, scaler = preprocess_data(googl_data)

    # Combine the data (if needed)
    X = np.concatenate((aapl_X, tsla_X, googl_X), axis=0)
    y = np.concatenate((aapl_y, tsla_y, googl_y), axis=0)

    return X, y, scaler

# Build the LSTM model
def build_lstm_model(X_train):
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))

    # Output layer
    model.add(Dense(units=1))  # Predict the next day's price

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function to load data, train the model, and save it
def main():
    # Load and preprocess data for multiple stocks
    X, y, scaler = load_and_preprocess_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build and train the LSTM model
    model = build_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save('models/lstm_model.h5')
    print("LSTM model has been trained and saved as 'lstm_model.h5'.")

if __name__ == '__main__':
    main()
