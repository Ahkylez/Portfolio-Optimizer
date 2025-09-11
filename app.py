import streamlit as st
import yfinance as yf
from scr.initialize_data import *
from scr.financial_math import *
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
import numpy as np

st.set_page_config(page_title="Simple Portfolio Optimizer", layout="wide")
st.title("Simple Portfolio Optimizer")

# Add 1 dollar back test
# compare to s&p 500


# Sidebar for properties
with st.sidebar:
    # Get Stock symbols from user
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
    st.divider()
    # Slider for number of portfolios
    number_of_portfolios = st.slider(
        "Number of portfolios to simulate", 100, 1000, 10000)

# So no errors appear when no data is entered
if not symbols:
    st.info("Add at least one valid ticker in the sidebar.")
    st.stop()

# Do Financial Math
data = fetch_prices(symbols)

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


# Return vs Volatility Chart
n = number_of_securites(expected_returns)
# number_of_portfolios = 10000 # make scalar
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



## Clean this
# ---- Combined bar chart: MVP vs Max Sharpe ----
max_sharpe_w = weight[maxIndex, :]

fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(n)
bar_w = 0.4

ax.bar(x - bar_w/2, w_mvp,       bar_w, label='MVP')
ax.bar(x + bar_w/2, max_sharpe_w, bar_w, label='Max Sharpe')

ax.set_xticks(x)
ax.set_xticklabels(symbols, rotation=45)
ax.set_ylabel("Weight")
ax.set_title("Portfolio Weights: MVP vs Max Sharpe")
ax.legend()
st.pyplot(fig)



plt.figure(figsize=(12,10))
plt.scatter(expectedVolatility, expectedReturn, c=sharpeRatio)
plt.xlabel("Expected Volatility")
plt.ylabel("Expected Log Returns")
plt.colorbar(label='SR')
plt.scatter(expectedVolatility[maxIndex],expectedReturn[maxIndex], c='red', label='Max Sharpe Point')

Sigma_np = Sigma.values if hasattr(Sigma, "values") else Sigma

mvp_vol = float(np.sqrt(w_mvp @ Sigma_np @ w_mvp))
mvp_ret = float(expected_log_returns @ w_mvp)

plt.scatter(mvp_vol, mvp_ret, marker='*', s=300, edgecolors='k', linewidths=1.2, label='MVP')
plt.legend()

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

# corr_return_matrix = returns.corr()


# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_return_matrix, cmap="coolwarm", center=0, annot=False, linewidths=0.5)

# plt.title("Stock Correlation Heatmap", fontsize=16)
# st.pyplot(plt)