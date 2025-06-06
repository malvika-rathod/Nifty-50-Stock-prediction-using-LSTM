import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features_and_labels(df, trend_window=5, return_threshold=0.01):
    df['Return'] = df['Close'].pct_change()
    df['LogClose'] = np.log(df['Close'])
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(1)

    # Future return over N days
    df['FutureReturn'] = df['Close'].shift(-trend_window) / df['Close'] - 1
    df['Direction'] = (df['FutureReturn'] > return_threshold).astype(int)

    # Drop uncertain or flat samples
    df = df[(df['FutureReturn'] > return_threshold) | (df['FutureReturn'] < -return_threshold)]

    df.dropna(inplace=True)
    return df

def preprocess_for_classification(df, sequence_length=60):
    df = add_features_and_labels(df)

    features = df[['LogClose', 'RSI', 'MACD', 'Momentum', 'Return']].values
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(df['Direction'].values[i])

    return np.array(X), np.array(y), scaler

def evaluate_classifier(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred) * 100,
        "recall": recall_score(y_true, y_pred) * 100,
        "f1": f1_score(y_true, y_pred) * 100
    }
