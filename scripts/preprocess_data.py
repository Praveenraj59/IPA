import pandas as pd
import numpy as np
import os

# Function to calculate Moving Average
def moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate RSI
def rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Preprocess the data
def preprocess_data(input_file, output_file):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load raw data
    data = pd.read_csv(input_file)

    # Ensure 'Close' column is numeric
    if 'Close' in data.columns:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Convert to numeric, set errors to NaN

    # Drop rows where 'Close' is NaN
    data.dropna(subset=['Close'], inplace=True)

    # Add technical indicators
    data['MA50'] = moving_average(data, 50)
    data['RSI14'] = rsi(data, 14)

    # Drop rows with NaN (resulting from rolling calculations)
    data.dropna(inplace=True)

    # Save the processed data
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(input_file='data/AAPL_data.csv', output_file='data/AAPL_processed_data.csv')
    preprocess_data(input_file='data/GOOGL_data.csv', output_file='data/GOOGL_processed.csv')
    preprocess_data(input_file='data/TSLA_data.csv', output_file='data/TSLA_processed.csv')

