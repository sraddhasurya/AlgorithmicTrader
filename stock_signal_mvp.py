import yfinance as yf
import pandas as pd

class StockSignalPipeline:
    def __init__(self, ticker, start_date, end_date, short_window=50, long_window=200):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.short_window = short_window
        self.long_window = long_window
        self.data = None

    def fetch_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return self.data

    def calculate_moving_averages(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data to calculate moving averages.")
        self.data['Short_MA'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['Long_MA'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        return self.data

    def generate_signals(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data to generate signals.")
        signals = []
        for i in range(len(self.data)):
            short_ma = self.data['Short_MA'].iloc[i]
            long_ma = self.data['Long_MA'].iloc[i]
            if short_ma > long_ma:
                signal = 'Buy'
            elif short_ma < long_ma:
                signal = 'Sell'
            else:
                signal = 'Hold'
            signals.append(signal)
        self.data['Signal'] = signals
        return self.data

    def run_pipeline(self):
        self.fetch_data()
        if self.data.empty:
            print(f"No data found for {self.ticker} in the given date range.")
            return None
        self.calculate_moving_averages()
        self.generate_signals()
        return self.data

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    pipeline = StockSignalPipeline(ticker, start_date, end_date)
    result = pipeline.run_pipeline()
    if result is not None:
        print("\nSignals:")
        print(result[['Close', 'Short_MA', 'Long_MA', 'Signal']]) 