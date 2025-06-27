import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class LSTMStockPredictor:

    #Parameters:
    #'ticker': the stock symbol to fetch data 
    #'seq_length=60': how many past days of data the model should look at
    #'hidden_size=50': number of neurons in the hidden layer of the LSTM model
    #'epochs=30': number of full passes over the training dataset
    #'lr=0.001': learning rate, how quickly the model updates weights during training
    def __init__(self, ticker, seq_length=60, hidden_size=512, epochs=150, lr=0.0005):
        self.ticker=ticker
        self.seq_length=seq_length
        self.hidden_size=hidden_size
        self.epochs=epochs
        self.lr=lr
        self.model=None
        #Scales input data to a range, allows neurals networks to perform better 
        self.scaler= MinMaxScaler()
        #CUDA is a much faster GPU but if not available use CPU
        if torch.backends.mps.is_available():
            self.device="mps"
        else:
            self.device="cpu"
        print("Using device:", self.device)


    #Gets data from the start to end date, returns a dataframe of only the close prices 
    def fetch_data(self,start="2015-01-01", end="2025-06-01"):
        df=yf.download(self.ticker, start=start, end=end)
        if df.empty:
            raise ValueError("No data fetched.")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    #Converts time series dtat into input/output sequences for training our LSTM 
    def create_sequences(self,data):
        #x holds input sequences and y holds target values 
        x,y,=[],[]
        for i in range(self.seq_length, len(data)):
            x.append(data[i-self.seq_length:i])
            y.append(data[i,3])
        return np.array(x), np.array(y)
    
    #Preprocesses raw price daya into a form that can be used to train our LSTM model
    def prepare_data(self, df):
        #Normalizes the data using nin-max scaling ((0,1) range)
        scaled_data=self.scaler.fit_transform(df)
        X,y= self.create_sequences(scaled_data)
        X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, shuffle=False)
        #Each set is converted to a torch.Tensor to be used in training and evaluation
        X_train=torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train=torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(self.device)
        X_test=torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test=torch.tensor(y_test.reshape(-1,1), dtype=torch.float32).to(self.device)
        
        return X_train, y_train, X_test, y_test
    
    #Defines the LSTM model we are using for the time series forecasting
    class LSTMModel(nn.Module):
        #Parameters:
        #input_size: number of input feaures per time step
        #hidden_size: number of features in the hidden state 
        #num_layers: number of stacked LSTM layers
        def __init__(self, input_size=5, hidden_size=512, num_layers=2):
            super().__init__()
            #defines a LSTM layer
            self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            #takes the last LSTM output of hidden_size and maps it to a single value
            self.fc=nn.Linear(hidden_size,1)

        #defines how input data flows through the model during prediction, returns the predicted values 
        def forward(self, x):
            out , _=self.lstm(x)
            out=self.fc(out[:, -1, :])
            return out
        
    #Trains an LSTM Model on the training dataset
    def train(self, X_train, y_train):
        self.model=self.LSTMModel(hidden_size=self.hidden_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size=64
        train_dataset=torch.utils.data.TensorDataset(X_train, y_train)
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        #repeats the training process for number of times defined by self.epochs
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss=0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                output = self.model(batch_X)
                loss = criterion(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
            avg_loss=epoch_loss/len(train_dataset)
            #prints current loss every 5 epochs to trach how training is going 
            if (epoch+1)%5==0:
                print(f"Epoch{epoch+1}/{self.epochs}, Loss:{loss.item():.4f}")

    #Evaluates the performance of the trained LSTM model on test data
    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            predictions=self.model(X_test).cpu().numpy()
            true=y_test.cpu().numpy()

            # Pad to match the 5D input shape expected by inverse_transform
            predictions_padded = np.zeros((predictions.shape[0], 5))
            predictions_padded[:, 3] = predictions[:, 0]  # only Close column predicted

            true_padded = np.zeros((true.shape[0], 5))
            true_padded[:, 3] = true[:, 0]

            # Inverse transform and extract Close only
            predictions = self.scaler.inverse_transform(predictions_padded)[:, 3]
            true = self.scaler.inverse_transform(true_padded)[:, 3]

            # train_size = int(len(df) * 0.8)
            # train_df = df[:train_size]
            # test_df = df[train_size:]
            # scaled_train = self.scaler.fit_transform(train_df)
            # scaled_full = self.scaler.transform(df) 
        return predictions, true
    
    #Runs the full pipeline of fetching data, preprocessing, training, and evaluation
    def run(self):
        print(f"Running LSTM model for {self.ticker}...")
        df=self.fetch_data()
        X_train, y_train, X_test, y_test= self.prepare_data(df)
        self.train(X_train, y_train)
        preds, true= self.evaluate(X_test, y_test)
        return preds, true

