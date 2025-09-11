import yfinance as yf
import pandas as pd

# get stock data

def fetch_prices(symbols: list[str], period='1y', interval="1d") -> pd.DataFrame:
    """
    Fetches histrocial closing prices for a list of stock symbols.

    Args:
        symbols (list[str]): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        period (str): The time period for the data (e.g., '1y', '6mo', 'max').
        interval (str): The data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        pd.DataFrame: A DataFrame containing the 'Close' prices for each symbol.
                      Returns an empty DataFrame if no data is found or an error occurs.
    """
    if not symbols:
        return pd.DataFrame() 
    
    try:

        stock_data = yf.download(tickers=symbols, period=period, interval=interval, progress=False)
        
        if stock_data.empty:
            print("No data found for given symbols")
            return pd.DataFrame()
        
        stock_df = pd.DataFrame(stock_data)
        price_df = stock_df["Close"].dropna()
        return price_df
    except:
        return pd.DataFrame()
