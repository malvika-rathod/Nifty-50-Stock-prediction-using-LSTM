import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def load_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

def load_custom_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    df['LogClose'] = np.log(df['Close'])
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(1)

    # Lagged versions
    df['RSI_lag1'] = df['RSI'].shift(1)
    df['MACD_lag1'] = df['MACD'].shift(1)
    df['Momentum_lag1'] = df['Momentum'].shift(1)

    df.dropna(inplace=True)
    return df

def preprocess_data(df, sequence_length=60):
    df = add_technical_indicators(df)

    features = df[['LogClose', 'MA20', 'RSI', 'MACD', 'Momentum', 'RSI_lag1', 'MACD_lag1', 'Momentum_lag1']].values
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict log(close)

    return np.array(X), np.array(y), scaler

def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return rmse, mape
