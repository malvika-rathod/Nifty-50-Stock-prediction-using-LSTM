import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
from utils import load_stock_data, preprocess_data, evaluate_model, load_custom_csv
from model_lstm import get_or_train_model
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils_classifier import preprocess_for_classification, evaluate_classifier, add_features_and_labels
from classifier_model import get_or_train_classifier

st.set_page_config(page_title="NIFTY-50 LSTM Predictor", layout="wide")
st.title("📈 LSTM Stock Price Predictor")

# ──────────────────────────────────────────────────────────────────────
# Data selection
input_mode = st.sidebar.radio("Data Source", ["NIFTY-50 (Yahoo)", "Upload CSV"])

if input_mode == "NIFTY-50 (Yahoo)":
    ticker = st.sidebar.selectbox("Select Ticker", ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS'])
    start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=730))
    end_date = st.sidebar.date_input("End Date", date.today())
    df = load_stock_data(ticker, start_date, end_date)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Date' and 'Close'", type=['csv'])
    if uploaded_file:
        df = load_custom_csv(uploaded_file)
        ticker = "custom"
    else:
        st.warning("Please upload a CSV to proceed.")
        st.stop()

# ──────────────────────────────────────────────────────────────────────
# Data Preview and Fixing for MultiIndex or wrong headers
st.subheader("📋 Raw Data Preview")

# Flatten multi-index headers if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# Fix columns if necessary
df.columns = [col.strip().capitalize() for col in df.columns]

if "Close" not in df.columns:
    possible_close = [col for col in df.columns if "close" in col.lower()]
    if possible_close:
        df.rename(columns={possible_close[0]: "Close"}, inplace=True)

if "Date" not in df.columns and df.index.name == "Date":
    df.reset_index(inplace=True)

st.dataframe(df.head())

# ──────────────────────────────────────────────────────────────────────
# Chart
if "Date" in df.columns and "Close" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    st.subheader("📊 Closing Price History")
    st.line_chart(df["Close"])
else:
    st.warning("⚠️ Data must contain 'Date' and 'Close' columns.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# LSTM Model
sequence_len = st.sidebar.slider("Sequence Length", 30, 100, 60)
X, y, scaler = preprocess_data(df, sequence_length=sequence_len)

split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

st.info("Training or loading LSTM model...")
# same as before — keep your existing version
# just make sure to update:
model = get_or_train_model(ticker, X_train, y_train, (X.shape[1], X.shape[2]))



predicted_scaled = model.predict(X_test)
dummy = np.zeros((predicted_scaled.shape[0], 8))  # now 8 features
dummy[:, 0] = predicted_scaled[:, 0]
log_predicted_prices = scaler.inverse_transform(dummy)[:, 0].reshape(-1, 1)
predicted_prices = np.exp(log_predicted_prices)  # reverse log

dummy_actual = np.zeros((y_test.shape[0], 8))
dummy_actual[:, 0] = y_test
log_actual_prices = scaler.inverse_transform(dummy_actual)[:, 0].reshape(-1, 1)
actual_prices = np.exp(log_actual_prices)
# ──────────────────────────────────────────────────────────────────────
# Evaluation
rmse, mape = evaluate_model(actual_prices, predicted_prices)
st.metric("📉 RMSE", f"{rmse:.2f}")
st.metric("📊 MAPE", f"{mape:.2f}%")

# ──────────────────────────────────────────────────────────────────────
# Plot Predicted vs Actual
st.subheader("🔮 Predicted vs Actual Prices")
fig = go.Figure()
fig.add_trace(go.Scatter(y=actual_prices.flatten(), name="Actual"))
fig.add_trace(go.Scatter(y=predicted_prices.flatten(), name="Predicted"))
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# Confusion Matrix (for price direction)
st.subheader("📌 Confusion Matrix (Price Direction: Up/Down)")
actual_direction = (actual_prices[1:] > actual_prices[:-1]).astype(int).flatten()
predicted_direction = (predicted_prices[1:] > predicted_prices[:-1]).astype(int).flatten()

cm = confusion_matrix(actual_direction, predicted_direction)
fig_cm, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(ax=ax)
st.pyplot(fig_cm)

# ──────────────────────────────────────────────────────────────────────
# Prediction Accuracy (threshold-based)
threshold_percent = st.sidebar.slider("Accuracy Threshold (%)", 1, 10, 3)
sequence_len = st.sidebar.slider("Sequence Length", 30, 100, 60, key="seq_len")
threshold_percent = st.sidebar.slider("Accuracy Threshold (%)", 1, 10, 3, key="threshold")

errors = np.abs(predicted_prices.flatten() - actual_prices.flatten())
tolerance = actual_prices.flatten() * (threshold_percent / 100)
within_threshold = errors <= tolerance
threshold_accuracy = (np.sum(within_threshold) / len(within_threshold)) * 100



# Directional accuracy
delta_actual = actual_prices[1:] - actual_prices[:-1]
delta_pred = predicted_prices[1:] - predicted_prices[:-1]

actual_direction = (delta_actual > 0).astype(int)
predicted_direction = (delta_pred > 0).astype(int)

mask = np.abs(delta_actual) > 0.001  # ignore near-flat movements
directional_accuracy = (actual_direction[mask] == predicted_direction[mask]).mean() * 100

# ──────────────────────────────────────────────────────────────────────
# Display on Streamlit
st.metric("✅ Threshold Accuracy", f"{threshold_accuracy:.2f}%")
st.metric("📈 Directional Accuracy", f"{directional_accuracy:.2f}%")

if threshold_accuracy >= 80:
    st.success(f"🎉 Excellent! Prediction accuracy is {threshold_accuracy:.2f}%")
else:
    st.warning(f"Accuracy is {threshold_accuracy:.2f}% — try training longer or adding features.")

# ──────────────────────────────────────────────────────────────────────
# CLASSIFIER SECTION WITH THRESHOLD FILTERING AND IMPROVEMENTS
df_labeled = add_features_and_labels(df)  # adds FutureReturn and Direction
threshold = threshold_percent / 100

# Filter by FutureReturn magnitude
confident_df = df_labeled[df_labeled['FutureReturn'].abs() > threshold]

if confident_df.empty:
    st.warning("⚠️ No data available after threshold filtering. Try lowering the threshold.")
else:
    # Add lagged indicators
    confident_df['RSI_lag1'] = confident_df['RSI'].shift(1)
    confident_df['MACD_lag1'] = confident_df['MACD'].shift(1)
    confident_df['Momentum_lag1'] = confident_df['Momentum'].shift(1)
    confident_df.dropna(inplace=True)

    # Reduce sequence length for better coverage
    X, y, scaler = preprocess_for_classification(confident_df, sequence_length=30)

    if X.shape[0] == 0:
        st.warning("⚠️ Not enough data to form sequences after filtering. Try lowering the threshold.")
    else:
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = get_or_train_classifier(ticker, X_train, y_train, (X.shape[1], X.shape[2]))
        y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)

        if len(y_test) == 0:
            st.warning("⚠️ No samples to evaluate after filtering.")
        else:
            metrics = evaluate_classifier(y_test, y_pred)
            st.metric("✅ Accuracy", f"{metrics['accuracy']:.2f}%")
            st.metric("🎯 Precision", f"{metrics['precision']:.2f}%")
            st.metric("🔁 Recall", f"{metrics['recall']:.2f}%")
            st.metric("📊 F1-Score", f"{metrics['f1']:.2f}%")

            st.subheader("Confusion Matrix (Filtered by Accuracy Threshold)")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
            disp.plot(ax=ax)
            st.pyplot(fig)

            st.subheader("Confusion Matrix (Filtered by Accuracy Threshold)")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
            disp.plot(ax=ax)
            st.pyplot(fig)
