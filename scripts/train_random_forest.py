import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load LSTM predictions and actual returns
def load_lstm_predictions_and_targets():
    # Example: Mock LSTM predictions and actual returns for demonstration
    # Replace with actual LSTM predictions for AAPL, TSLA, and GOOGL
    lstm_predictions = {
        "AAPL": np.random.rand(100),
        "TSLA": np.random.rand(100),
        "GOOGL": np.random.rand(100)
    }
    actual_returns = {
        "AAPL": np.random.rand(100),
        "TSLA": np.random.rand(100),
        "GOOGL": np.random.rand(100)
    }

    # Combine predictions into a feature matrix (rows: samples, columns: stocks)
    X = np.column_stack((lstm_predictions["AAPL"], lstm_predictions["TSLA"], lstm_predictions["GOOGL"]))

    # Combine actual returns into a single target array (average returns)
    y = (actual_returns["AAPL"] + actual_returns["TSLA"] + actual_returns["GOOGL"]) / 3

    return X, y

# Train the Random Forest model
def train_random_forest(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error on test data: {mae:.4f}")

    return rf_model

# Main function to train and save the Random Forest model
def main():
    # Load LSTM predictions and corresponding actual returns
    X, y = load_lstm_predictions_and_targets()

    # Train the Random Forest model
    rf_model = train_random_forest(X, y)

    # Save the trained Random Forest model
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("Random Forest model has been saved as 'random_forest_model.pkl'.")

if __name__ == '__main__':
    main()
