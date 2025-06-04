from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

def build_classifier(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_or_train_classifier(ticker, X_train, y_train, input_shape, model_dir="models/"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{ticker}_cls.h5"

    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            model.predict(np.zeros((1, *input_shape)))
            return model
        except:
            os.remove(model_path)

    model = build_classifier(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stop, checkpoint]
    )
    return model
