from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error')
    return model

def get_or_train_model(ticker, X_train, y_train, input_shape, model_dir="models/"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{ticker}.h5"

    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            model.predict(np.zeros((1, *input_shape)))
            return model
        except:
            os.remove(model_path)

    model = create_lstm_model(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stop, checkpoint]
    )
    return model

