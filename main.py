import pandas as pd
import numpy as np
from numpy.linalg import inv
import yfinance as yf

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


st.title("Simple Portfolio Optimizer")

with st.sidebar:
    st.header("Stock Symbols")
    symbols = []
    myTickers = st.text_input("Enter stock tickers eg. AAPL NVDA...")

    ticker_list = myTickers.replace(",", " ").split()
    for t in ticker_list:
        try:
            ticker = yf.Ticker(t)
            info = ticker.info  
            if info and info.get("regularMarketPrice") is not None:
                symbols.append(t)
        except Exception as e:
            st.write(f"Error fetching {t}: {e}")



# get stock data
#symbols = ['AAPL', 'NVDA', 'ACI', 'FTFT', 'INBK']
stock_data = yf.download(tickers=symbols, period='1y')
stock_df = pd.DataFrame(stock_data)
price_df = stock_df["Close"].dropna()


# Calculating the daily returns throughout the year
returns = price_df.pct_change().dropna() # K_i

# Creating the expected return matrix, using a historical sample
expected_return = np.array(returns.mean()) 

# Get covarience of return matrix
cov_return_matrix = returns.cov().to_numpy()

# Get weights
n = np.shape(expected_return)[0] # amount of securites
u = np.ones(n, dtype=int) # allows me to get feasible set

inv_cov_return_matrix = inv(cov_return_matrix)
num = u @ inv_cov_return_matrix
den = u @ inv_cov_return_matrix @ u.T
w_mvp = num/den

mvp_df = pd.DataFrame({
    "Stock": symbols,
    "Daily Return": expected_return,
    "MVP Weight": w_mvp
})

# weight must sum up to one
if (not np.isclose(sum(w_mvp), 1)):
    raise("Weight is nowhere near 1")

#print(mvp_df.round(4))


st.write(mvp_df.round(4))

# Plot weights
plt.figure(figsize=(8,5))
plt.bar(mvp_df["Stock"], mvp_df["MVP Weight"])
plt.xlabel("Stock")
plt.ylabel("MVP Weight")
plt.title("Minimum Variance Portfolio Weights")
plt.xticks(rotation=45)
st.pyplot(plt)




#print(myTickers)




corr_return_matrix = returns.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr_return_matrix, cmap="coolwarm", center=0, annot=False, linewidths=0.5)

plt.title("Stock Correlation Heatmap", fontsize=16)
st.pyplot(plt)

