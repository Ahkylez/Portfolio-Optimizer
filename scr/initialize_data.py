import yfinance as yf
import pandas as pd

# get stock data
#symbols = ['AAPL', 'NVDA', 'ACI', 'FTFT', 'INBK']
def get_symbol_close_data(symbols):
    stock_data = yf.download(tickers=symbols, period='1y')
    stock_df = pd.DataFrame(stock_data)
    price_df = stock_df["Close"].dropna()
    return price_df