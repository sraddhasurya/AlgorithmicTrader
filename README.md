# Stock Forecasting and Signal Generation System

## Overview

This project combines multiple machine learning and statistical approaches to forecast stock prices and generate actionable buy/sell signals. It includes:
an LSTM model for next-day price and direction prediction,
a GBM (Geometric Brownian Motion) simulation for uncertainty modeling,
a moving average crossover-based signal pipeline,
a test suite to validate signal logic.

## Project Structure

### LSTM Trading Model
**lstm_trading_model.py** -- uses a Long Short-Term Memory (LSTM) neural network to forecast both future closing prices and the direction of price movement for a given stock (e.g., AAPL), incorporates momentum as a technical indicator and is trained on historical data from Yahoo Finance

  **Features:**
    
  - Predicts the next-day closing price using LSTM regression
  - Classifies price direction one day ahead of (binary classification)
  - Uses price difference to improve the model signal
  - Uses a dual-head architecture (regression and classification)
  - Tracks training loss
  - Evaluates performance with MAE, RMSE, and classification accuracy

    ![Screenshot 2025-07-02 at 5 08 11 PM](https://github.com/user-attachments/assets/e7859e39-100a-41cb-b26d-253f658bab0e)

    This graph shows the LSTM model's ability to track short-term price trends on test data. While the predicted closing prices (blue) closely follow the overall movement of actual prices (orange), the model struggles to capture sudden directional shifts, showing the challenge of precise financial forecasting, even when general trend prediction is relatively accurate.


  **What I learned:**

  This project taught me about applying ML to time series data in reak-world finance context. I first learned how to prepare stock market data effectively. I used yfinance to pull historical prices, engineered my own features like momentum, and scaled inputs using MinMaxScaler. I learned how to create sequences for LSTM input and label them for both regression (next price) and classification (direction). Designing the model helped me undertand how LSTMs work under the hood and how to build multi-headed architectures in PyTorch to predict price and direction. I learned how to combine loss functions (MSELoss for price and BCEWithLogitsLoss for direction) and how to weight the classification loss to handle class imbalance. 

  A big lession I learned came from the evaluation process. The model achieved very low MAE and RMSE on closing price predictions, showing that it learned the genral price trends, but the classification results were very weak. The model defaulted to predicting "up" and achieved an accuracy of 42%, which is worse than just guessing. This helped teach me that accurate price prediction doesn't guarantee relaible trading signals and it requires strong, more discriminative features than regression. These failures showed me the limitations in simple models and the need to use external signals other than just price. 

  **Problems:**

  - Direction bias: The model learned to always predict up probably due to noisy signals, loss imbalance (price MSE < direction BCE) and insufficient LSTM depth or features
  - Precision collapse: Overfitting: The model performed well on the training price data but lacked generalization power for classification

  **Next Steps:** adding more technical indicators, experimenting with multi-layer LSTMs, etc

### GBM Stock Predictor
gbm_stock_predictor.py-- implements a Monte Carlo simulations using Geometric Brownian Motion (GBM) to model and forecast future stock price paths, this tool is built on core principles from quantitative finance and is intended for educational, exploratory, and light research use

  **Features:**

  - Gets historical close prices from Yahoo Finance using yfinance
  - Calculates the drift (μ) and volatility (σ) from historical log returns
  - Simulates a price trajectory over a given number of future days using GBM
  - Runs a Monte Carlo simulation to generate hundreds of possible future price paths
  - Reports the expected return and volatility from the similations

    <img width="998" alt="Screenshot 2025-07-02 at 5 11 25 PM" src="https://github.com/user-attachments/assets/11085256-000b-4186-8537-0290fb15294c" />

    This chart is a single simulated 30-day price path for AAPL using the GBM model. It shows the stochastic natures of price movements and the type of behavior typical in financial markets. 
<img width="1198" alt="Screenshot 2025-07-02 at 5 11 45 PM" src="https://github.com/user-attachments/assets/f2f24fa0-c949-4ad1-b23a-a29dc95e35b9" />

This shows 500 simulated stock price paths for AAPL over 30 days using a Monte Carlo simulation based on the GBM model. The red line is the mean prediction across all simulations, showing the expected trajectory and range of future outcomes. 


 **What I learned:**

 This project was my introduction to stochastic processes and how they are applied to real world financial data. I learned several quantitative finance concepts. I used the Geometric Brownian Motion equation which is a differential equation that assumes stock prices follow a continuous time random walk with a drift and volatility component. I estimated the drift and volatility by using the histroical mean and standard deviation of log returns. I used this equation to generate many possible futures of a stock which mirrors how traders evaluate risk by exploring distributions of outcomes (Monte Carlo Simulation). I learned why log returns are preferred over simple returns due to them being time additive and mathematically more convenient when computing drift and volatility. The simulation coputed the ecpected return, standard deviation, and the confidence intervals for price forecasts. 

 **Problems:**

 This model assumes costant volatility and drift with isn't realistic due to market conditions always changing. Models like GARCH or regime-switching models could offer better forecats. This model does not account for jumps, news shocks, or volatility clustering. 

 **Next Steps:** adding percentile bands and VaR calculations for more advance risk analysis, including options pricing using these simulations. 

### Basic Stock Signal  

**stock_signal_mvp.py**-- implements a simple trading signal generator using moving average crossovers, uses histroical stock data to compute short and long term moving averages and generates the signals based on techincal patterns
**test_stock_signal_mvp.py**--contains the unit tests for the StockSignalPipeline class in stock_signal_mvp.py, verfies the correctness of key components 

**Features:**

- Pulls historical stock data from Yahoo Finance using yfinance
- User can specifiy short and long moving average windows (default: 50 and 200 days)
- Generates trading sgnal based on crossover between short and long term moving averages
- Outputs date, stock price, short/long MA, and signal
- Tests each component separatley and validates expected properties 

  **What I Learned:**
  I used the technical indicators used by traders to identify momentum and learned how the simple moving average crossovers can indicate upward(bullish) or downward (bearish) trends. I learned how to translate price trands into discrete "Buy", "Sell", and "Hold" decisions using conditionals and vectorized operations in pandas.

  **Problems:**
  This model is purely rule-based and is not predictive. The logic is based soley on SMA crossovers and doesn't take into account any other real-world factors.

  **Next Steps:**
  Integrate with a different kind fo model like an LSTM model for hybrid signal and price prediction and adding more indicators


  ## Conclusion:
  This project was my first exploration of quantitative finance, time series prediction, and signal generation using both statistical and machine learning techniques. I build an LSTM based neural netwrok to predict stock prices and direct, a GBM based Monte Carlo simulation to model price uncertainy, and a technical indicator based signal pipeline using moving averages. Along the way, I learned how to clean and engineer financial data, apply deep learning to noisy time series, and interpret simulation outputs to estimate market risk. My biggest takeaway was learning learning that accurate price prediction doesn't mean good trading performance, especially when the model struggles to classify direction. I also learned how models like GBM offer insights into uncertainty but rely on assumptions that always hold in markets. I now have a better understanding of both potential and limitations of forecasting in finance. I am excited to explore more advances stratgeies like hybrid models, volatility modeling, and risk-based decision-making in the future.

    

  

