import yfinance as yf

# Function to download stock data from Yahoo Finance
def get_stock_data(ticker):
    stock_data = yf.download(ticker, start="2019-01-01", end="2024-01-01")
    return stock_data

# Example usage
aapl_data = get_stock_data("AAPL")
tsla_data = get_stock_data("TSLA")
googl_data = get_stock_data("GOOGL")

# Save the data to CSV (Optional, if you need to store it)
aapl_data.to_csv("data/AAPL_data.csv")
tsla_data.to_csv("data/TSLA_data.csv")
googl_data.to_csv("data/GOOGL_data.csv")
