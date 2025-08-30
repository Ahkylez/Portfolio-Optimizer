import pandas as pd
import numpy as np
from numpy.linalg import inv
import yfinance as yf

import matplotlib.pyplot as plt
import seaborn as sns


symbols = ['AAPL', 'NVDA', 'ACI', 'FTFT', 'INBK']
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


# Format for readability
print(mvp_df.round(4))



corr_return_matrix = returns.corr()

# corr_matrix is your DataFrame of correlations())()()()
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_return_matrix, cmap="coolwarm", center=0, annot=False, linewidths=0.5)

# plt.title("Stock Correlation Heatmap", fontsize=16)
# plt.show()

