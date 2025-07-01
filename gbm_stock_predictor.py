import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class GBMStockPredictor:
    """
    A class to model stock price movement using Geometric Brownian Motion (GBM),
    a stochastic process used in quantitative finance to simulate price paths.
    """

    def __init__(self, ticker):
        """
        Initializes the GBMStockPredictor with the stock ticker.
        """
        self.ticker = ticker         # Ticker symbol (e.g., "AAPL", "SPY")
        self.mu = None               # Expected return (annualized drift)
        self.sigma = None            # Volatility (annualized)
        self.df = None               # Historical closing price data
        self.S0 = None               # Most recent closing price
        self.simulated_price = None  # Last simulated price path

    def fetch_data(self, period='1y'):
        """
        Downloads historical price data from Yahoo Finance.
        Only uses the 'Close' column.

        Args:
            period (str): Time range for historical data (e.g., '1y', '6mo').

        Returns:
            pd.Series: Close prices over the requested period.
        """
        df = yf.download(self.ticker, period=period)
        if df.empty:
            raise ValueError("No data fetched.")
        self.df = df['Close']
        self.S0 = float(self.df.iloc[-1])  # Most recent price
        return self.df

    def calculate_parameters(self):
        """
        Calculates the GBM parameters: drift (mu) and volatility (sigma),
        based on historical log returns.

        Returns:
            (mu, sigma): Tuple of annualized drift and volatility.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        log_returns = np.log(self.df / self.df.shift(1)).dropna()
        self.mu = float((log_returns.mean() * 252).iloc[0])             # Annualized return
        self.sigma = float((log_returns.std() * np.sqrt(252)).iloc[0])  # Annualized volatility
        return self.mu, self.sigma

    def simulate_future_prices(self, days=30):
        """
        Simulates a single future price path using GBM over a number of days.

        Args:
            days (int): Number of trading days to simulate.

        Returns:
            np.ndarray: Simulated price series.
        """
        if self.mu is None or self.sigma is None:
            self.calculate_parameters()

        T = 1 / 252      # Time period (1 trading day)
        N = days
        dt = T / N       # Time step
        Z = np.random.normal(0, 1, N)  # Random normal variables
        S = np.zeros(N)
        S[0] = self.S0

        for t in range(1, N):
            S[t] = S[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z[t])

        self.simulated_price = S
        return S

    def simulate_multiple_paths(self, days=60, simulations=1000):
        """
        Simulates multiple GBM paths for Monte Carlo analysis.

        Args:
            days (int): Number of future days to simulate.
            simulations (int): Number of simulation paths.

        Returns:
            np.ndarray: 2D array of shape (simulations, days).
        """
        if self.mu is None or self.sigma is None:
            self.calculate_parameters()

        dt = 1 / 252
        S0 = self.S0
        paths = np.zeros((simulations, days))
        paths[:, 0] = S0

        for t in range(1, days):
            Z = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
            )
        return paths

    def plot_prediction(self):
        """
        Plots the most recent single simulated GBM price path.
        """
        if self.simulated_price is None:
            raise ValueError("Run simulate() first.")

        plt.figure(figsize=(10, 5))
        plt.plot(self.simulated_price, label="Simulated Prices")
        plt.title(f"{self.ticker} Price Simulation (GBM)")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_monte_carlo_simulation(self, paths):
        """
        Plots multiple Monte Carlo simulation paths and their mean.

        Args:
            paths (np.ndarray): 2D array of simulated price paths.
        """
        plt.figure(figsize=(12, 6))
        for i in range(min(100, paths.shape[0])):  # Limit number of paths to avoid overplotting
            plt.plot(paths[i], color='gray', alpha=0.2)

        mean_path = paths.mean(axis=0)
        plt.plot(mean_path, color='red', label="Mean Prediction", linewidth=2)

        plt.title(f"Monte Carlo Simulation for {self.ticker}")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.show()

    def predict_next_day(self, simulations=1000):
        """
        Predicts the next day's price using Monte Carlo GBM simulation.

        Args:
            simulations (int): Number of simulations to average over.

        Returns:
            dict: Contains current price, predicted price, predicted direction, and confidence interval.
        """
        if self.mu is None or self.sigma is None:
            self.calculate_parameters()

        dt = 1 / 252
        S0 = self.S0
        next_day_prices = []

        for _ in range(simulations):
            Z = np.random.normal()
            S_next = S0 * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)
            next_day_prices.append(S_next)

        next_day_prices = np.array(next_day_prices)
        predicted_price = np.mean(next_day_prices)
        direction = 1 if predicted_price > S0 else 0

        lower_bound = np.percentile(next_day_prices, 2.5)
        upper_bound = np.percentile(next_day_prices, 97.5)

        return {
            "current_price": S0,
            "predicted_price": predicted_price,
            "direction": "up" if direction == 1 else "down",
            "confidence_interval": (lower_bound, upper_bound)
        }

    def evaluate_direction_accuracy(self, days=30, simulations=1000):
        """
        Evaluates how often the predicted price direction matches the actual direction
        over the past `days` by walking backward through historical data.

        Args:
            days (int): Number of past days to evaluate on.
            simulations (int): Number of simulations per prediction.
        """
        correct = 0
        price_errors = []
        close_prices = self.df.values

        for i in range(-days - 1, -1):  # Indexes into past `days` using negative indexing
            window = self.df.iloc[:i]   # Use only data up to the current point

            if len(window) < 2:
                continue  # Skip if insufficient data

            self.df = window
            self.S0 = float(window.iloc[-1].item())  # Most recent price in the window

            try:
                self.calculate_parameters()
            except Exception:
                continue

            simulated_paths = self.simulate_multiple_paths(days=1, simulations=simulations)
            predicted_price = simulated_paths[:, -1].mean()
            actual_price = close_prices[i + 1]

            predicted_direction = 1 if predicted_price > self.S0 else 0
            actual_direction = 1 if actual_price > self.S0 else 0

            if predicted_direction == actual_direction:
                correct += 1

            price_errors.append(abs(predicted_price - actual_price))

        accuracy = correct / days
        mae = np.mean(price_errors)

        print(f"Direction Accuracy over last {days} days: {accuracy * 100:.2f}%")
        print(f"Mean Absolute Error (MAE) of predicted prices: ${mae:.2f}")

if __name__ == "__main__":
    ticker = "AAPL"  # Specify the stock ticker symbol (e.g., Apple Inc.)

    # Create an instance of the GBMStockPredictor for the specified ticker
    gbm = GBMStockPredictor(ticker)

    # Fetch historical stock price data (default = past 1 year)
    gbm.fetch_data(period="1y")

    # Calculate GBM model parameters (drift μ and volatility σ)
    gbm.calculate_parameters()

    # Simulate a single future price path for the next 30 trading days
    single_path = gbm.simulate_future_prices(days=30)

    # Plot this single simulation path to visualize price trajectory
    gbm.plot_prediction()

    # Run Monte Carlo simulation with 1000 paths over 30 days
    mc_paths = gbm.simulate_multiple_paths(days=30, simulations=1000)

    # Plot many simulation paths along with the average path
    gbm.plot_monte_carlo_simulation(mc_paths)

    # Compute and print 95% confidence interval for the 30th day
    final_prices = mc_paths[:, -1]  # Final predicted price from each path
    lower = np.percentile(final_prices, 2.5)
    upper = np.percentile(final_prices, 97.5)
    print(f"95% confidence interval after 30 days: ${lower:.2f} - ${upper:.2f}")

    # Evaluate how often the model correctly predicts the direction (up/down) of price 
    # changes over the past 30 days using simulated forecasts
    gbm.evaluate_direction_accuracy(days=30, simulations=1000)
