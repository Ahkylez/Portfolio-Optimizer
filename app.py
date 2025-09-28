import streamlit as st
import yfinance as yf
from scr.financial_math import *
import matplotlib.pyplot as plt
import seaborn as sns


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
        "Number of portfolios to simulate", 100, 100000, 2500)
    
    st.divider()
    allow_short_selling_mvp = st.checkbox("Allow Short Selling for MVP", value=False)
    

# So no errors appear when no data is entered
if not symbols:
    st.info("Add at least one valid ticker in the sidebar.")
    st.stop()

# Do math
po = PortfolioOptimizer(symbols)
w_mvp = po.calculate_optimized_mvp(allow_short_selling_mvp)
w_max_sharpe, expectedReturn, expectedVolatility, sharpeRatio, maxIndex = po.compute_sharpe_ratio_weights(number_of_portfolios)
returns_ef, volatility_opt = po.calulate_efficient_frontier(expectedReturn)

# Metrics
st.subheader("Portfolio Metrics")

col1, col2 = st.columns(2)
mvp_annual_return = po.annualized_return(w_mvp)
mvp_annual_risk = po.portfolio_risk_annualized(w_mvp)

max_sharpe_annual_return = po.annualized_return(w_max_sharpe)
max_sharpe_annual_risk = po.portfolio_risk_annualized(w_max_sharpe)

with col1:
    st.markdown("Minimum Volatility Portolio")
    st.metric(label="Annualized Return", value=f"{mvp_annual_return:.2%}")
    st.metric(label="Annualized Risk", value=f"{mvp_annual_risk:.2%}")

with col2: 
    st.markdown("Max Sharpe Portfolio")
    st.metric(label="Annualized Return", value=f"{max_sharpe_annual_return:.2%}")
    st.metric(label="Annualized Risk", value=f"{max_sharpe_annual_risk:.2%}")

# Weights to purchase

st.subheader("Optimal Portfolio Weights")

weights_df = pd.DataFrame({
    "MVP Weight": w_mvp,
    "Max Sharpe Weight": w_max_sharpe
}, index=symbols) # Use the stock symbols as the index

weights_df_formatted = weights_df.style.format("{:.2%}")
st.dataframe(weights_df_formatted, use_container_width=True)


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

## Main graph

plt.figure(figsize=(12,10))
plt.scatter(expectedVolatility, expectedReturn, c=sharpeRatio, cmap='viridis', label='Simulated Portfolios')
plt.xlabel("Expected Volatility (Standard Deviation)")
plt.ylabel("Expected Log Returns")
plt.colorbar(label='Sharpe Ratio')
plt.title("Portfolio Optimization with Efficient Frontier")

Sigma_np = po.sigma.values if hasattr(po.sigma, "values") else po.sigma
mvp_vol = float(np.sqrt(w_mvp @ Sigma_np @ w_mvp))
mvp_ret = float(po.expected_log_returns @ w_mvp)

# Plot Max Sharpe Point
plt.scatter(expectedVolatility[maxIndex],expectedReturn[maxIndex], c='red', marker='*', s=300, edgecolors='k', linewidths=1.2, label='Max Sharpe Point')

# Plot MVP
plt.scatter(mvp_vol, mvp_ret, c='green', marker='X', s=200, edgecolors='k', linewidths=1.2, label='Minimum Volatility Portfolio (MVP)')

# Plot the Efficient Frontier
# Filter out NaNs if any optimization failed
valid_ef_indices = ~np.isnan(volatility_opt)
plt.plot(volatility_opt[valid_ef_indices], returns_ef[valid_ef_indices], linestyle='--', color='black', linewidth=2, label='Efficient Frontier')

plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
st.pyplot(plt)



results = po.out_of_sample_backtest()

if results: # Check if the backtest was successful
    plt.figure(figsize=(12, 6))
    
    results['MVP_Value'].plot(label='Minimum Volatility Portfolio (MVP)', linewidth=2)
    results['MaxSharpe_Value'].plot(label='Maximum Sharpe Ratio Portfolio', linewidth=2)
    results['SP500_Value'].plot(label='S&P 500', linewidth=2)

    plt.title('Out-of-Sample Portfolio Backtest (Starting Value: $1)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)
