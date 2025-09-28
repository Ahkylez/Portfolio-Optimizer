import pandas as pd
import numpy as np
from numpy.linalg import inv
import yfinance as yf
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, symbols: list[str], trading_periods=252, prices: pd.DataFrame | None = None):
        self.symbols = symbols
        self.trading_periods = trading_periods

        self.prices = self._fetch_prices()
        self.returns, self.log_returns = self._calculate_returns()

        self.prices = prices if prices is not None else self._fetch_prices()

        self.expected_returns = self.returns.mean().to_numpy()
        self.expected_log_returns = self.log_returns.mean().to_numpy()

        self.n = len(self.symbols)
        self.cov_return_matrix = self._get_cov_matrix(self.returns)
        
        self.sigma = self.log_returns.cov()

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, trading_periods=252):
        # Infer symbols from columns
        symbols = list(prices.columns)

        # Call the normal constructor, but inject these symbols and this price data
        return cls(symbols=symbols, trading_periods=trading_periods, prices=prices)
        
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


    def _calculate_returns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    
    # Efficient Frontier Calculations
    def negativeSR(self, w):
        w = np.array(w)
        # Ensure expected_log_returns is a 1D array or Series
        R = np.sum(self.expected_log_returns.values * w) if hasattr(self.expected_log_returns, 'values') else np.sum(self.expected_log_returns * w)
        Sigma_np = self.sigma.values if hasattr(self.sigma, "values") else self.sigma
        V = np.sqrt(np.dot(w.T, np.dot(Sigma_np,w)))
        SR = R/V
        return -1 * SR
    
    def checkSumToOne(self, w):
        return np.sum(w) - 1

    def minimizeMyVolatility(self, w):
        w = np.array(w)
        Sigma_np = self.sigma.values if hasattr(self.sigma, "values") else self.sigma
        V = np.sqrt(np.dot(w.T, np.dot(Sigma_np,w)))
        return V

    def getReturn(self, w):
        w = np.array(w)
        # Ensure expected_log_returns is a 1D array or Series
        R = np.sum(self.expected_log_returns.values * w) if hasattr(self.expected_log_returns, 'values') else np.sum(self.expected_log_returns * w)
        return R
    
    # Calculate Efficient Frontier points
    def calulate_efficient_frontier(self, expectedReturn):

        w0 = np.full(self.n, 1.0/self.n)
        bounds = ((0, 1),) * self.n
        constraints = ({'type':'eq', 'fun':self.checkSumToOne})

        # Determine min and max possible returns from simulated portfolios to set the range for the efficient frontier
        min_return = expectedReturn.min()
        max_return = expectedReturn.max()
        returns_ef = np.linspace(min_return * 0.9, max_return * 1.1, 50) # Extend range slightly
        volatility_opt = []

        for R_target in returns_ef:
            constraints = ({'type':'eq', 'fun':self.checkSumToOne},
                        {'type':'eq', 'fun': lambda w: self.getReturn(w) - R_target})
            opt = minimize(self.minimizeMyVolatility, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if opt.success:
                volatility_opt.append(opt['fun'])
            else:
                volatility_opt.append(np.nan) # Append NaN if optimization fails

        # Convert to numpy array for plotting
        volatility_opt = np.array(volatility_opt)
        return returns_ef, volatility_opt
    
    def calculate_optimized_mvp(self, allow_short_selling: bool = False) -> np.ndarray:
        w0 = np.full(self.n, 1.0 / self.n)
        if allow_short_selling:
            bounds = ((-1.0, 1.0),) * self.n
        else:
            bounds = ((0, 1),) * self.n

        constraints = ({'type': 'eq', 'fun': self.checkSumToOne})
        
        opt = minimize(self.minimizeMyVolatility, w0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints)
        
        if opt.success:
            return opt.x
        else:
            print("MVP optimization failed. Returning equal weights.")
            return w0

    # Get weights, explain math later in doc string
    def calculate_mvp(self) -> np.ndarray:
        u = np.ones(self.n) # allows me to get feasible set
        inv_cov_return_matrix = inv(self.cov_return_matrix)
        num = u @ inv_cov_return_matrix
        den = u @ inv_cov_return_matrix @ u.T
        w_mvp = num/den
        return w_mvp
    
    def compute_sharpe_ratio_weights(self, number_of_portfolios) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weight = np.zeros((number_of_portfolios, self.n))
        expectedReturn = np.zeros(number_of_portfolios)
        expectedVolatility = np.zeros(number_of_portfolios)
        sharpeRatio = np.zeros(number_of_portfolios)

        # Use self.sigma directly
        Sigma = self.sigma

        for k in range(number_of_portfolios):
            # generate random weight vector
            w = np.array(np.random.random(self.n))
            w = w / np.sum(w)
            weight[k,:] = w
            #expected log returns
            # Use expected_log_returns directly
            expectedReturn[k] = np.sum(self.expected_log_returns * w) 
            #expected volitility
            expectedVolatility[k] = np.sqrt(w.T @ Sigma @ w)
            # Sharpe Ratio
            sharpeRatio[k] = expectedReturn[k]/expectedVolatility[k]

        maxIndex = sharpeRatio.argmax()
        w_max_sharpe = weight[maxIndex, :]
        # Return all the simulated data and the Max Sharpe weights
        return w_max_sharpe, expectedReturn, expectedVolatility, sharpeRatio, maxIndex
    
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

        # Backtest data
    def _simulate_portfolio_value(self, weights: np.ndarray, testing_prices: pd.DataFrame, initial_investment: float = 1.0) -> pd.Series:
       
        # Calculate daily simple returns for the testing period
        testing_returns = testing_prices.pct_change().dropna()
        
        # Calculate the daily portfolio return
        daily_portfolio_returns = pd.Series(
            testing_returns.values @ weights,
            index=testing_returns.index
        )
        
        # Calculate the cumulative returns
        cumulative_returns = (1 + daily_portfolio_returns).cumprod()
        
        # Calculate the portfolio value
        portfolio_value = initial_investment * cumulative_returns
        
        return portfolio_value


    def out_of_sample_backtest(self, training_days: int = 60, initial_investment: float = 1.0) -> dict[str, pd.Series]:

        df_prices = self._fetch_prices(period='2y') # Get 2 years of data
        if df_prices.empty:
            print("Empty DF: Could not fetch enough data for backtest.")
            return {}
        
        if len(df_prices) < training_days * 1.5:
            print(f"Not enough data for backtest. Need at least {training_days} training days and some testing days.")
            return {}

        # Split Data
        training_prices = df_prices.iloc[:training_days]
        testing_prices = df_prices.iloc[training_days:]
        
        # Train the Optimizer on the training data
        po_train = PortfolioOptimizer.from_prices(training_prices, trading_periods=self.trading_periods)

        # Calculate Weights
        w_mvp_opt = po_train.calculate_optimized_mvp(allow_short_selling=False)
        w_sharpe, _, _, _, _ = po_train.compute_sharpe_ratio_weights(number_of_portfolios=5000)

        results = {}
        
        # MVP Simulation
        mvp_value_series = self._simulate_portfolio_value(w_mvp_opt, testing_prices, initial_investment)
        results['MVP_Value'] = mvp_value_series
        
        # Max Sharpe Simulation
        sharpe_value_series = self._simulate_portfolio_value(w_sharpe, testing_prices, initial_investment)
        results['MaxSharpe_Value'] = sharpe_value_series
        

        return results