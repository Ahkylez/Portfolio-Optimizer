import streamlit as st
import yfinance as yf
from scr.initialize_data import *
from scr.financial_math import *
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
import numpy

st.title("Simple Portfolio Optimizer")


# Get user input for stock ticker
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

# So no errors appear when no data is entered
if not symbols:
    st.info("Add at least one valid ticker in the sidebar.")
    st.stop()

# Do Financial Math
data = get_symbol_close_data(symbols)
returns = daily_returns(data)
log_returns = log_returns_from_prices(data)

expected_returns = expected_returns(returns)
expected_log_returns = expected_log_returns(log_returns)

cov_return_matrix = cov_return_matrix(returns)
w_mvp = calculate_mvp(expected_returns, cov_return_matrix)

port_return = portfolio_return(w_mvp, expected_returns)
port_log_return = portfolio_log_return(w_mvp, expected_log_returns)
port_risk = portfolio_risk(w_mvp, cov_return_matrix)

# Make table
mvp_df = pd.DataFrame({
    "Stock": symbols,
    "Daily Return": expected_returns,
    "Daily Log Return": expected_log_returns,
    "MVP Weight": w_mvp
})
st.write(mvp_df.round(4))

st.write(f'Portfolio Dailys Return: {port_return}')
st.write(f'Portfolio Dailys Log Return: {port_log_return}')
st.write(f'Portfolio Risk: {port_risk}')

# Plot weights
plt.figure(figsize=(8,5))
plt.bar(mvp_df["Stock"], mvp_df["MVP Weight"])
plt.xlabel("Stock")
plt.ylabel("MVP Weight")
plt.title("Minimum Variance Portfolio Weights")
plt.xticks(rotation=45)
st.pyplot(plt)

# Return vs Volatility Chart
n = number_of_securites(expected_returns)
number_of_portfolios = 10000 # make scalar
weight = np.zeros((number_of_portfolios, n))
expectedReturn = np.zeros(number_of_portfolios)
expectedVolatility = np.zeros(number_of_portfolios)
sharpeRatio = np.zeros(number_of_portfolios)

Sigma = log_returns.cov()

for k in range(number_of_portfolios):
    # generate random weight vector
    w = np.array(np.random.random(n))
    w = w / np.sum(w)
    weight[k,:] = w
    #expected log returns
    expectedReturn[k] = np.sum(expected_log_returns * w)
    #expected volitility
    expectedVolatility[k] = np.sqrt(np.dot(w.T, np.dot(Sigma,w)))
    # Sharpe Ratio
    sharpeRatio[k] = expectedReturn[k]/expectedVolatility[k]

maxIndex = sharpeRatio.argmax()
weight[maxIndex,:]

plt.figure(figsize=(12,10))
plt.scatter(expectedVolatility, expectedReturn, c=sharpeRatio)
plt.xlabel("Expected Volatility")
plt.ylabel("Expected Log Returns")
plt.colorbar(label='SR')
plt.scatter(expectedVolatility[maxIndex],expectedReturn[maxIndex], c='red')
st.pyplot(plt)


def negativeSR(w):
    w = np.array(w)
    R = np.sum(expected_log_returns * weight)
    V = np.sqrt(np.dot(w.T, np.dot(Sigma,w)))
    SR = R/V
    return -1 * SR

def checkSumToOne(w):
    return np.sum(w) - 1

w0 = np.full(n, 1.0/n)
bounds = ((0, 1),) * n
constraints = ({'type':'eq', 'fun':checkSumToOne})
w_opt = minimize(negativeSR, w0, method='SLSQP', bounds=bounds, constraints=constraints)

returns = np.linspace(0,0.07,50) # make variable 0.07
volatility_opt = []
def minimizeMyVolatility(w):
    w = np.array(w)
    V = np.sqrt(np.dot(w.T, np.dot(Sigma,w)))
    return V
def getReturn(w):
    w = np.array(w)
    R = np.sum(expected_log_returns * w)
    return R

for R in returns:
    constraints = ({'type':'eq', 'fun':checkSumToOne},
                   {'type':'eq', 'fun': lambda w: getReturn(w) - R})

    opt = minimize(minimizeMyVolatility, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    volatility_opt.append(opt['fun'])



corr_return_matrix = returns.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr_return_matrix, cmap="coolwarm", center=0, annot=False, linewidths=0.5)

plt.title("Stock Correlation Heatmap", fontsize=16)
st.pyplot(plt)