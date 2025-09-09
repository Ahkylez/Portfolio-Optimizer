import pandas as pd
import numpy as np
from numpy.linalg import inv

# Calculating the daily returns throughout the year
def daily_returns(price_df):
    return price_df.pct_change().dropna() # K_i

# Creating the expected return matrix, using a historical sample
def expected_returns(returns):
    return np.array(returns.mean()) 

# Get covarience of return matrix
def cov_return_matrix(returns):
    return returns.cov().to_numpy()

def test_mvp(w_mvp):
    if (not np.isclose(sum(w_mvp), 1)):
        raise("Weight is nowhere near 1")

def number_of_securites(expected_return):
    return np.shape(expected_return)[0]

# Get weights
def calculate_mvp(expected_return, cov_return_matrix):
    n = np.shape(expected_return)[0] # amount of securites
    u = np.ones(n, dtype=int) # allows me to get feasible set

    inv_cov_return_matrix = inv(cov_return_matrix)
    num = u @ inv_cov_return_matrix
    den = u @ inv_cov_return_matrix @ u.T
    w_mvp = num/den
    test_mvp(w_mvp)
    return w_mvp

def portfolio_return(w_mvp, expected_return):
    return w_mvp @ expected_return.T

def portfolio_log_return(w_mvp, expected_log_return):
    return w_mvp @ expected_log_return.T


def portfolio_risk(w_mvp, cov_return_matrix):
    return w_mvp @ cov_return_matrix @ w_mvp.T


def log_returns_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df).diff().dropna()


def expected_log_returns(log_returns):
    return log_returns.mean().to_numpy()



# Sharpe Ratio


# Return vs volatility
