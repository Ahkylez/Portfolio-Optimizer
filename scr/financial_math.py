import pandas as pd
import numpy as np
from numpy.linalg import inv

# Calculating the daily returns throughout the year
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Given stock price data frame, return data frame of the daily return or the percent change: 
    (Current Value - Previous Value) / Previous Value. 

    Args:
        prices (pd.DataFrame): Data frame of stock prices

    Return: 
        pd.DataFrame: Data frame of daily return. With all NAN removed.
    """
    return prices.pct_change().dropna() # K_i

def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Given stock price data frame, return data frame of the daily log return or the percent change: 
    (Log(Current Value) - log(Previous Value)) / log(Previous Value). 

    Args:
        prices (pd.DataFrame): Data frame of stock prices

    Return: 
        pd.DataFrame: Data frame of daily log return. With all NAN removed.
    """
    return np.log(prices).diff().dropna()

# Creating the expected return and log return matrix, using a historical sample
def expected_returns(returns: pd.DataFrame, annualize: bool = False, trading_periods: int = 252) -> np.ndarray: 
    """
    Calculates the expected returns from a DataFrame of historical returns.

    Args:
        returns (pd.DataFrame): A DataFrame where each column represents the 
                                historical returns of an asset.
        annualize (bool): If True, annualizes the returns. Defaults to False.
        trading_periods (int): The number of trading periods in a year. 
                               Used for annualization. Defaults to 252 for daily data.

    Returns:
        np.ndarray: A NumPy array of the mean (expected) returns for each asset.
    """
    mean_returns = returns.mean()

    if annualize:
        mean_returns *= trading_periods

    return np.array(mean_returns)
        
def expected_log_returns(log_returns: pd.DataFrame, annualize: bool = False, trading_periods: int = 252) -> np.ndarray:
    """
    Calculates the expected (mean) log returns for a portfolio of assets.

    Args:
        log_returns (pd.DataFrame): A DataFrame where each column represents the 
                                    historical log returns of an asset.
        annualize (bool): If True, annualizes the returns. Defaults to False.
        trading_periods (int): The number of trading periods in a year. 
                               Used for annualization. Defaults to 252 for daily data.

    Returns:
        np.ndarray: A NumPy array containing the mean log return for each asset.
    """

    mean_log_returns = log_returns.mean()
    
    if annualize:
        mean_log_returns *= trading_periods
    
    return mean_log_returns.to_numpy()

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







# Sharpe Ratio


# Return vs volatility
