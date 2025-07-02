import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

class LSTMStockPredictor:
    """
    Implements an LSTM-based deep learning model to predict stock closing prices and their future directional movement (up/down).
    The model uses historical close prices and a momentum indicator to forecast:
      - The next closing price (regression)
      - The direction of price movement in the next few days (classification)
    """
    def __init__(self, ticker, seq_length=30, hidden_size=128, epochs=100, lr=0.001):
        # Set hyperparameters and initialize key model attributes
        self.ticker = ticker
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use Apple M1 GPU if available
        print("Using device:", self.device)

    def fetch_data(self, start="2015-01-01", end="2025-06-01"):
        """
        Downloads historical stock data (Close price) and adds a momentum feature (price difference).
        Drops initial NaN value created by the `.diff()` operation.
        """
        df = yf.download(self.ticker, start=start, end=end)
        if df.empty:
            raise ValueError("No data fetched.")
        df = df[['Close']].copy()
        df['Momentum'] = df['Close'].diff()  # Momentum = today's close - yesterday's close
        df.dropna(inplace=True)  # Drop first row with NaN momentum
        return df

    def create_sequences(self, data):
        """
        Constructs sequences of `seq_length` timesteps as input.
        Each target contains:
          - the current price at time t (for regression)
          - a binary label (1 if price at t+3 > price at t, else 0)
        """
        x, y, direction = [], [], []
        for i in range(self.seq_length, len(data) - 3):
            x.append(data[i - self.seq_length:i])
            y.append(data[i, 0])  # Predict current price
            direction.append(1 if data[i + 1, 0] > data[i, 0] else 0)  # Price direction after 1 day
        return np.array(x), np.array(y), np.array(direction)

    def prepare_data(self, df):
        """
        Scales features and splits dataset into training and test sets.
        Converts the data into PyTorch tensors for use in the LSTM model.
        Also prints class balance for diagnostics.
        """
        scaled_data = self.scaler.fit_transform(df.values)
        # Store price column min and scale for inverse transform
        self.price_min = self.scaler.data_min_[0]
        self.price_scale = self.scaler.scale_[0]
        X, y_price, y_dir = self.create_sequences(scaled_data)
        X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = train_test_split(
            X, y_price, y_dir, test_size=0.1, shuffle=False)

        # Convert data into PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_price_train = torch.tensor(y_price_train.reshape(-1, 1), dtype=torch.float32).to(self.device)
        y_dir_train = torch.tensor(y_dir_train.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_price_test = torch.tensor(y_price_test.reshape(-1, 1), dtype=torch.float32).to(self.device)
        y_dir_test = torch.tensor(y_dir_test.reshape(-1, 1), dtype=torch.float32).to(self.device)

        # Print class distribution for both test and train sets
        print("Class balance (test set):")
        print("Down days:", int(y_dir_test.sum().item()), "/ Up days:", len(y_dir_test) - int(y_dir_test.sum().item()))

        print("Class balance (train set):")
        print("Down days:", int(y_dir_train.sum().item()), "/ Up days:", len(y_dir_train) - int(y_dir_train.sum().item()))

        return X_train, y_price_train, y_dir_train, X_test, y_price_test, y_dir_test

    def inverse_transform_price(self, scaled_price):
        """
        Inverse transforms a scaled price value using the scaler's min and scale for the price column only.
        """
        return scaled_price * self.price_scale + self.price_min

    class LSTMModel(nn.Module):
        def __init__(self, input_size=2, hidden_size=128, num_layers=1):
            # Defines the LSTM model with two heads (price and direction)
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc_price = nn.Linear(hidden_size, 1)   # Regression head
            self.fc_dir = nn.Linear(hidden_size, 1)     # Classification head

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # Take the last time step's hidden state
            price = self.fc_price(out)
            direction = self.fc_dir(out)
            return price, direction

    def train(self, X_train, y_price_train, y_dir_train):
        """
        Trains the LSTM model using both price (MSE) and direction (weighted BCE) losses.
        Uses class balancing via pos_weight to address class imbalance in direction prediction.
        """
        self.model = self.LSTMModel(input_size=X_train.shape[2], hidden_size=self.hidden_size).to(self.device)
        criterion_price = nn.MSELoss()

        # Compute class weights for binary classification
        num_positive = y_dir_train.sum().item()
        num_negative = len(y_dir_train) - num_positive
        if num_positive == 0:
            pos_weight = torch.tensor([1.0]).to(self.device)
        else:
            pos_weight = torch.tensor([num_negative / num_positive]).to(self.device)
        pos_weight = torch.clamp(pos_weight, min=0.5, max=3.0)
        criterion_dir = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = 128
        train_dataset = torch.utils.data.TensorDataset(X_train, y_price_train, y_dir_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_X, batch_y_price, batch_y_dir in train_loader:
                pred_price, pred_dir_logits = self.model(batch_X)
                pred_dir_logits = torch.clamp(pred_dir_logits, min=-10, max=10)  # Prevent extreme values
                loss_price = criterion_price(pred_price, batch_y_price)
                loss_dir = criterion_dir(pred_dir_logits, batch_y_dir)
                loss = loss_price + loss_dir  

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(train_dataset)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Total Loss: {loss.item():.4f}, Price Loss: {loss_price.item():.4f}, Direction Loss: {loss_dir.item():.4f}")

    def evaluate(self, X_test, y_price_test):
        """
        Predicts closing prices on test data and rescales predictions to original price range.
        Returns predicted and actual prices.
        """
        self.model.eval()
        with torch.no_grad():
            pred_price, _ = self.model(X_test)
            pred_price = pred_price.cpu().numpy()
            true = y_price_test.cpu().numpy()

            # Inverse transform only the price column
            pred_price = self.inverse_transform_price(pred_price[:, 0])
            true = self.inverse_transform_price(true[:, 0])
        return pred_price.flatten(), true.flatten()

    def evaluate_accuracy(self, X_test, y_price_test, y_dir_test):
        """
        Evaluates both regression and classification performance.
        - Computes MAE and RMSE for price prediction.
        - Computes accuracy for direction prediction.
        """
        self.model.eval()
        with torch.no_grad():
            pred_price, pred_dir_logits = self.model(X_test)
            pred_dir = torch.sigmoid(pred_dir_logits)  # Convert logits to probabilities

            # Unscale predicted prices
            pred_price_np = pred_price.cpu().numpy()
            y_price_np = y_price_test.cpu().numpy()
            pred_price_rescaled = self.inverse_transform_price(pred_price_np[:, 0])
            true_price_rescaled = self.inverse_transform_price(y_price_np[:, 0])

            # Metrics
            mae = np.mean(np.abs(pred_price_rescaled - true_price_rescaled))
            rmse = np.sqrt(np.mean((pred_price_rescaled - true_price_rescaled) ** 2))
            pred_dir_labels = (pred_dir.cpu().numpy() > 0.5).astype(int)
            true_dir_labels = y_dir_test.cpu().numpy().astype(int)
            acc = np.mean(pred_dir_labels == true_dir_labels)

        print(f"Direction Accuracy: {acc * 100:.2f}%")
        print(f"Price MAE: {mae:.4f}")
        print(f"Price RMSE: {rmse:.4f}")

        return acc, mae, rmse

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots a confusion matrix and classification report for the predicted directions.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Up", "Down"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Up", "Down"]))

if __name__ == "__main__":
    # Instantiate model
    predictor = LSTMStockPredictor("AAPL", seq_length=30, epochs=100)

    # Fetch and preprocess data
    df = predictor.fetch_data()
    X_train, y_price_train, y_dir_train, X_test, y_price_test, y_dir_test = predictor.prepare_data(df)

    # Train the model
    predictor.train(X_train, y_price_train, y_dir_train)

    # Evaluate regression performance
    preds, true = predictor.evaluate(X_test, y_price_test)

    # Evaluate classification and price error metrics
    acc, mae, rmse = predictor.evaluate_accuracy(X_test, y_price_test, y_dir_test)

    # Predict directions on test set and plot confusion matrix
    with torch.no_grad():
        _, dir_logits = predictor.model(X_test)
        dir_probs = torch.sigmoid(dir_logits)
        dir_preds = (dir_probs.cpu().numpy() > 0.5).astype(int)
        true_dirs = y_dir_test.cpu().numpy().astype(int)

    plot_confusion_matrix(true_dirs, dir_preds)

    # Plot predicted vs actual close prices
    plt.plot(preds, label="Predicted Close")
    plt.plot(true, label="Actual Close")
    plt.legend()
    plt.title("LSTM Predicted vs Actual Close Prices")
    plt.show()
