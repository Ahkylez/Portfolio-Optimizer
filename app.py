import streamlit as st
import yfinance as yf
from scr.financial_math import *
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
import numpy as np

st.set_page_config(page_title="Simple Portfolio Optimizer")
st.title("Simple Portfolio Optimizer")

# Add 1 dollar back test
# compare to s&p 500


# Sidebar for properties
with st.sidebar:
    # Get Stock symbols from user
    st.header("Stock Symbols")
    symbols = []
    
    # Get ticker from user
    myTickers = st.text_input("Enter stock tickers eg. AAPL NVDA...")
    ticker_list = myTickers.replace(",", " ").split()
    
    # Validate User inputed ticker
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
        "Number of portfolios to simulate", 100, 5000, 2500)

# So no errors appear when no data is entered
if not symbols:
    st.info("Add at least one valid ticker in the sidebar.")
    st.stop()

# Do math
po = PortfolioOptimizer(symbols)
w_mvp = po.calculate_mvp()
w_max_sharpe = po.compute_sharpe_ratio_weights(number_of_portfolios)



# Plots

fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(po.n)
bar_w = 0.4

ax.bar(x - bar_w/2, w_mvp,       bar_w, label='MVP')
ax.bar(x + bar_w/2, w_max_sharpe, bar_w, label='Max Sharpe')

ax.set_xticks(x)
ax.set_xticklabels(symbols, rotation=45)
ax.set_ylabel("Weight")
ax.set_title("Portfolio Weights: MVP vs Max Sharpe")
ax.legend()
st.pyplot(fig)




