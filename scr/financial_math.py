import pandas as pd
import numpy as np
from numpy.linalg import inv
import yfinance as yf


class PortfolioOptimizer:
    def __init__(self, symbols: list[str], trading_periods=252):
        self.symbols = symbols
        self.trading_periods = trading_periods

        self.prices = self._fetch_prices()

        self.returns, self.log_returns = self._calculate_all_returns()



        self.expected_returns = self.returns.mean().to_numpy()
        self.expected_log_returns = self.log_returns.mean().to_numpy()

        self.n = len(self.symbols)
        self.cov_return_matrix = self._get_cov_matrix(self.returns)
        
    def _fetch_prices(self, period='1y', interval="1d") -> pd.DataFrame:
        """
        Fetches histrocial closing prices for a list of stock symbols.

        Args:
            symbols (list[str]): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
            period (str): The time period for the data (e.g., '1y', '6mo', 'max').
            interval (str): The data interval (e.g., '1d', '1wk', '1mo').

        Return:
            pd.DataFrame: A DataFrame containing the 'Close' prices for each symbol.
                        Returns an empty DataFrame if no data is found or an error occurs.
        """
        if not self.symbols:
            return pd.DataFrame() 
        
        try:

            stock_data = yf.download(tickers=self.symbols, period=period, interval=interval, progress=False)
            
            if stock_data.empty:
                print("No data found for given symbols")
                return pd.DataFrame()
            
            stock_df = pd.DataFrame(stock_data)
            price_df = stock_df["Close"].dropna()
            return price_df
        except:
            return pd.DataFrame()


    def _calculate_all_returns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        simple_returns = self.prices.pct_change().dropna()
        log_returns = np.log(self.prices).diff().dropna()
        
        # align index so dates do not mismatch
        common_index = simple_returns.index.intersection(log_returns.index)

        simple_returns = simple_returns.loc[common_index]
        log_returns = log_returns.loc[common_index]
        
        return simple_returns, log_returns

    # Get covarience of matrix
    def _get_cov_matrix(self, returns) -> np.ndarray:
        return returns.cov().to_numpy()

    # Get weights, explain math later in doc string
    def calculate_mvp(self) -> np.ndarray:
        u = np.ones(self.n) # allows me to get feasible set
        inv_cov_return_matrix = inv(self.cov_return_matrix)
        num = u @ inv_cov_return_matrix
        den = u @ inv_cov_return_matrix @ u.T
        w_mvp = num/den
        return w_mvp
    
    def compute_sharpe_ratio_weights(self, number_of_portfolios):
        weight = np.zeros((number_of_portfolios, self.n))
        expectedReturn = np.zeros(number_of_portfolios)
        expectedVolatility = np.zeros(number_of_portfolios)
        sharpeRatio = np.zeros(number_of_portfolios)

        Sigma = self.log_returns.cov()

        for k in range(number_of_portfolios):
            # generate random weight vector
            w = np.array(np.random.random(self.n))
            w = w / np.sum(w)
            weight[k,:] = w
            #expected log returns
            expectedReturn[k] = np.sum(self.expected_log_returns * w)
            #expected volitility
            expectedVolatility[k] = np.sqrt(w.T @ Sigma @ w)
            # Sharpe Ratio
            sharpeRatio[k] = expectedReturn[k]/expectedVolatility[k]

        maxIndex = sharpeRatio.argmax()
        max_sharpe_w = weight[maxIndex, :]
        print(f"maxIndex, {maxIndex}/n")
        print(f"max_sharpe_w[0], {max_sharpe_w}")
        return max_sharpe_w






    # ---- Get portfolio metrics ----
    def portfolio_return(self, weights: np.ndarray):
        return weights @ self.expected_returns.T

    def portfolio_log_return(self, weights: np.ndarray):
        return weights @ self.expected_log_returns.T

    def annualized_return(self, weights: np.ndarray):
        daily_return = self.portfolio_log_return(weights)
        annualized_return = daily_return * self.trading_periods
        return annualized_return
    
    def portfolio_risk(self, weights: np.ndarray):
        return weights @ self.cov_return_matrix @ weights.T
    
    def portfolio_risk_annualized(self, weights: np.ndarray):
        portfolio_variance = self.portfolio_risk(weights)
        portfolio_std_dev_daily = np.sqrt(portfolio_variance)
        annualized_std_dev = portfolio_std_dev_daily * np.sqrt(self.trading_periods)
        return annualized_std_dev







