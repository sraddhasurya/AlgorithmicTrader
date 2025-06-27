import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from lstm_trading_model import LSTMStockPredictor  # <- update this to your class file name if needed

# --- Set the Ticker ---
ticker = "TSLA"  # change this to any valid stock symbol

# --- Instantiate and Run the Model ---
model = LSTMStockPredictor(ticker)
preds, true = model.run()

# --- Evaluation Metrics ---
rmse = np.sqrt(mean_squared_error(true, preds))
mae = mean_absolute_error(true, preds)
print(f"\n📊 Evaluation Metrics for {ticker}:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

# --- Plot Predictions vs Actual Prices ---
plt.figure(figsize=(12, 6))
plt.plot(true, label="Actual Prices", linewidth=2)
plt.plot(preds, label="Predicted Prices", linestyle="--")
plt.title(f"LSTM Stock Price Prediction for {ticker}")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
