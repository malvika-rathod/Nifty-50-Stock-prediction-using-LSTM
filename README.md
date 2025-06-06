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
## Screenshots
![image](https://github.com/user-attachments/assets/2592a1b4-b8a6-40d3-850d-ebfdd301fcb0)
-Raw data and closing price history.
![image](https://github.com/user-attachments/assets/6f2df0a8-63f6-491b-828d-a0a8ac5c40a5)
-RMSE, MAPE, Predicted vs Actual prices.
![image](https://github.com/user-attachments/assets/9536a1bf-2a72-4710-b45a-62edb20ab308)
-Sidebar.
![image](https://github.com/user-attachments/assets/954365b1-3635-437f-9904-b329f35b78d4)
-Threshold Accuracy for Accuracy Threshold(%) 3.
![image](https://github.com/user-attachments/assets/c7c59ef7-522a-494f-8dea-e6e77f92812c)
-Threshold Accuracy for Accuracy Threshold(%) 5.
![image](https://github.com/user-attachments/assets/e7d55c68-2bb6-43ee-9ee5-f7b3deab53e8)
-Multiple Tickers.


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
