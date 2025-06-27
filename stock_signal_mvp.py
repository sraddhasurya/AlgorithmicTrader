import yfinance as yf
import pandas as pd

#This class is used to fetch the data from the yfinance API, calculate the moving averages, and generate the signals
class StockSignalPipeline:
    #Initializes the class with the ticker, start date, end date, short window, and long window
    def __init__(self, ticker, start_date, end_date, short_window=50, long_window=200):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.short_window = short_window
        self.long_window = long_window
        self.data = None

    #Fetches the data from the yfinance API
    def fetch_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return self.data

    #Calculates the moving averages
    def calculate_moving_averages(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data to calculate moving averages.")
        #Calculates the short moving average using the close price and the short window
        self.data['Short_MA'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        #Calculates the long moving average using the close price and the long window
        self.data['Long_MA'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        return self.data

    #Generates the signals
    def generate_signals(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data to generate signals.")
        #Creates an empty list to store the signals
        signals = []
        #Loops through the data
        for i in range(len(self.data)):
            #Gets the short moving average
            short_ma = self.data['Short_MA'].iloc[i]
            #Gets the long moving average
            long_ma = self.data['Long_MA'].iloc[i]
            #Checks if the short moving average is greater than the long moving average
            if short_ma > long_ma:
                signal = 'Buy'
            #If the short moving average is less than the long moving average, the signal is 'Sell'
            elif short_ma < long_ma:
                signal = 'Sell'
            #If the short moving average is equal to the long moving average, the signal is 'Hold'
            else:
                signal = 'Hold'
            #Adds the signal to the list
            signals.append(signal)
        #Adds the signals to the data
        self.data['Signal'] = signals
        return self.data

    #Runs the pipeline
    def run_pipeline(self):
        self.fetch_data()
        #Checks if the data is empty
        if self.data.empty:
            print(f"No data found for {self.ticker} in the given date range.")
            #Returns None if the data is empty
            return None
        #Calculates the moving averages
        self.calculate_moving_averages()
        #Generates the signals
        self.generate_signals()
        #Returns the data
        return self.data

#Main function
if __name__ == "__main__":
        #Gets the ticker from the user
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    #Gets the start date from the user
    start_date = input("Enter start date (YYYY-MM-DD): ")
    #Gets the end date from the user
    end_date = input("Enter end date (YYYY-MM-DD): ")
    #Creates a new instance of the StockSignalPipeline class
    pipeline = StockSignalPipeline(ticker, start_date, end_date)
    #Runs the pipeline
    result = pipeline.run_pipeline()
    #Checks if the result is not None
    if result is not None:
        #Prints the signals
        print("\nSignals:")
        #Prints the data with the close price, short moving average, long moving average, and signal
        print(result[['Close', 'Short_MA', 'Long_MA', 'Signal']]) 