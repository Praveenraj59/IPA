import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import io
import base64

# Function to preprocess data for LSTM
def preprocess_data(data, time_step=60):
    close_data = data['Close'].values
    close_data = close_data.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_data = scaler.fit_transform(close_data)
    
    # Prepare data for LSTM (X: past 60 days, Y: next day's price)
    X, y = [], []
    for i in range(time_step, len(close_data)):
        X.append(close_data[i-time_step:i, 0])
        y.append(close_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Shape for LSTM (samples, time_steps, features)
    
    return X, y, scaler

# Function to predict using LSTM
def predict_with_lstm(X, model_path="models/lstm_model.h5"):
    model = load_model(model_path)
    return model.predict(X)

# Function to predict portfolio performance using Random Forest
import joblib

# Function to predict portfolio performance using Random Forest
def predict_with_random_forest(lstm_predictions, model_path="models/random_forest_model.pkl"):
    """
    Uses the trained Random Forest model to predict portfolio performance
    based on LSTM predictions.
    """
    model = joblib.load(model_path)

    # Use only the first 3 predictions if the model was trained with 3 features
    if len(lstm_predictions) > 3:
        lstm_predictions = lstm_predictions[:3]

    # Predict using the Random Forest model
    rf_prediction = model.predict([lstm_predictions])  # Expecting a 2D array
    return rf_prediction[0]

# Function to generate plot (visualization)
def generate_plot(stock_names, lstm_preds, rf_pred):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot LSTM predictions for each stock
    ax.bar(stock_names, lstm_preds, color='blue', alpha=0.7, label='LSTM Predictions')

    # Add Random Forest overall prediction as a line
    ax.axhline(y=rf_pred, color='red', linestyle='--', label='Portfolio Prediction')

    # Title and labels
    ax.set_title('Stock Predictions and Portfolio Outlook')
    ax.set_xlabel('Stocks')
    ax.set_ylabel('Predicted Returns')
    ax.legend()

    # Save the plot as a base64-encoded PNG
    img = io.BytesIO()
    plt.savefig(img, format='png')  # Save plot to a byte stream
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')  # Convert to base64 string
    plt.close(fig)  # Close the figure to avoid GUI interaction
    return plot_url

# Function to generate insights based on predictions
def generate_insights(rf_pred, stocks, lstm_predictions):
    # General message based on portfolio prediction
    if rf_pred > 0.6:
        outlook = "strong positive"
    elif rf_pred > 0.3:
        outlook = "moderate"
    else:
        outlook = "cautious"
    
    # Specific advice for each stock based on its prediction
    stock_advice = []
    for i, stock in enumerate(stocks):
        predicted_return = lstm_predictions[i]
        if predicted_return > 0.5:
            stock_advice.append(f"Consider increasing your investment in {stock['name']} as it has a predicted return of {predicted_return:.2f}.")
        elif predicted_return == 0:
            stock_advice.append(f"Consider reviewing your position in {stock['name']}. Its predicted return is neutral (0.00).")
        else:
            stock_advice.append(f"Be cautious with {stock['name']}. Its predicted return is low ({predicted_return:.2f}).")

    # Generate the final insights message
    if outlook == "strong positive":
        advice = f"The model suggests a {outlook} outlook for your portfolio. You may want to increase exposure to high-performing stocks like {stocks[0]['name']}."
    elif outlook == "moderate":
        advice = "The model suggests a moderate outlook. Consider maintaining your current investments, but keep an eye on the market."
    else:
        advice = "The model suggests a cautious outlook. You might want to consider diversifying into safer investments or reducing exposure to riskier stocks."

    return {"outlook": advice, "stock_advice": stock_advice}
