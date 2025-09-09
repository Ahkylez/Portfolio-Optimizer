import streamlit as st
import yfinance as yf
from scr.initialize_data import *
from scr.financial_math import *
import matplotlib.pyplot as plt
import seaborn as sns

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
expected_returns = expected_returns(returns)
cov_return_matrix = cov_return_matrix(returns)
w_mvp = calculate_mvp(expected_returns, cov_return_matrix)

# Make table
mvp_df = pd.DataFrame({
    "Stock": symbols,
    "Daily Return": expected_returns,
    "MVP Weight": w_mvp
})
st.write(mvp_df.round(4))

# Plot weights
plt.figure(figsize=(8,5))
plt.bar(mvp_df["Stock"], mvp_df["MVP Weight"])
plt.xlabel("Stock")
plt.ylabel("MVP Weight")
plt.title("Minimum Variance Portfolio Weights")
plt.xticks(rotation=45)
st.pyplot(plt)

corr_return_matrix = returns.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr_return_matrix, cmap="coolwarm", center=0, annot=False, linewidths=0.5)

plt.title("Stock Correlation Heatmap", fontsize=16)
st.pyplot(plt)