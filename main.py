import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import ta

def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_model_comparison():
    # Data Processing and Feature Engineering
    df = pd.read_csv("usdjpy_daily.csv", index_col="date", parse_dates=True)
    df['target'] = df['close'].shift(-1)

    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()

    # Add lagged and moving average features
    for i in range(1, 4):
        df[f'lag_{i}'] = df['close'].shift(i)
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()

    df = df.dropna()

    # Model Training and Evaluation
    features = [
        'lag_1', 'lag_2', 'lag_3', 'ma_7', 'ma_30',
        'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low'
    ]
    X = df[features]
    y = df['target']

    # Split data for traditional models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Support Vector Machine": SVR()
    }

    results = {}

    print("\nTraditional Model Comparison")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        results[name] = mae
        print(f"{name} - MAE: {mae:.4f}")

    # LSTM Model
    print("\nLSTM Model Training and Evaluation")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # LSTM Hyperparameters to tune
    sequence_length = 5  # Number of past days to consider for prediction
    lstm_units = 100       # Number of LSTM units (can be increased/decreased)
    epochs = 200           # Number of training epochs (can be increased/decreased)
    batch_size = 32       # Batch size for training (can be changed)

    X_lstm, y_lstm = create_sequences(X_scaled, sequence_length)
    y_lstm = y_scaled[sequence_length:]

    # Split data for LSTM
    train_size_lstm = int(len(X_lstm) * 0.8)
    X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
    y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

    lstm_model = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        LSTM(lstm_units, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse') # Consider other optimizers like 'RMSprop'

    history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

    lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
    lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled)
    lstm_mae = mean_absolute_error(scaler_y.inverse_transform(y_test_lstm), lstm_predictions)
    results["LSTM"] = lstm_mae
    print(f"LSTM - MAE: {lstm_mae:.4f}")

    # Results Summary
    best_model_name = min(results, key=results.get)
    best_model_mae = results[best_model_name]

    print("\n--- Overall Model Comparison ---")
    for name, mae in results.items():
        print(f"{name}: {mae:.4f}")
    print(f"\nBest performing model: {best_model_name} (MAE: {best_model_mae:.4f})")

    # Final Prediction with Best Model
    if best_model_name == "LSTM":
        last_sequence_scaled = scaler_X.transform(X.iloc[-sequence_length:])
        last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, X.shape[1])
        next_day_prediction_scaled = lstm_model.predict(last_sequence_scaled)
        next_day_prediction = scaler_y.inverse_transform(next_day_prediction_scaled)[0][0]
    else:
        best_model = models[best_model_name]
        last_row = X.iloc[-1:].copy()
        next_day_prediction = best_model.predict(last_row)[0]

    print(f"\nPredicted close for the next trading day (using {best_model_name}): {next_day_prediction:.4f}")

if __name__ == "__main__":
    run_model_comparison()
