# 📈 NIFTY-50 Stock Trend Predictor using LSTM

An interactive Streamlit app for predicting stock trends using LSTM-based time-series forecasting with directional evaluation and threshold-based accuracy control.

---

## 🚀 Features

- **LSTM-based Regression** to forecast future closing prices.
- **Directional Accuracy Evaluation** (Up/Down movement).
- **Threshold Accuracy Slider** to filter predictions based on tolerance.
- **Custom CSV Upload** or Yahoo Finance ticker support.
- **Advanced Technical Indicators**:
  - RSI, MACD, Momentum
  - Moving Average (MA20)
  - Lagged versions for better signal
- **Confusion Matrix and Metrics** for directional evaluation:
  - Accuracy, Precision, Recall, F1-Score

---

## 📊 Key Accomplishments

- Implemented **threshold-based accuracy evaluation**.
- Integrated **technical and lagged indicators** into the forecasting pipeline.
- Allowed **interactive sequence length tuning** for time-series input.
- Structured app to support both **Yahoo Finance** and **user-uploaded data**.
- Visualized performance via **line charts** and **confusion matrices**.

---

## 🧠 Model

- `model_lstm.py`: LSTM model for log-transformed closing price prediction.

---

## 📂 Tech Stack

- **Frontend**: Streamlit
- **Backend/ML**: TensorFlow, Keras, Scikit-learn
- **Data Source**: Yahoo Finance or user-uploaded CSV

---

## 📝 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
