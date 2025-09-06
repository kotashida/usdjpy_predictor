# USD/JPY Exchange Rate Forecasting

## Project Overview

This project develops and evaluates a suite of machine learning models to forecast the daily closing price of the USD/JPY foreign exchange pair. The primary objective is to apply and compare various statistical and deep learning techniques, identifying the most accurate model for short-term prediction. The analysis involves robust feature engineering, model training, and a rigorous, quantitative evaluation of performance based on Mean Absolute Error (MAE).

The project demonstrates a systematic approach to time-series forecasting, from data acquisition and preprocessing to model implementation and validation. The final output provides a prediction for the next trading day's closing price, underpinned by a clear, data-driven model selection process.

## Methodology

The forecasting methodology is structured to ensure a robust and impartial comparison of traditional statistical models and a more complex neural network architecture.

### 1. Data Acquisition and Feature Engineering

- **Data Source:** Historical daily USD/JPY exchange rate data (open, high, low, close) was programmatically fetched from the Alpha Vantage API.
- **Feature Creation:** To enhance the predictive power of the models, the initial dataset was augmented with a variety of engineered features. This critical step transforms the raw time-series data into a richer feature space for the learning algorithms.
    - **Technical Indicators:** Standard financial market indicators were calculated to capture market momentum, trend, and volatility. These include:
        - **Relative Strength Index (RSI):** A 14-day RSI was used to identify overbought or oversold conditions.
        - **Moving Average Convergence Divergence (MACD):** The MACD, its signal line, and the histogram were included to gauge the direction and momentum of the trend.
        - **Bollinger Bands:** Upper and lower bands were computed to measure market volatility.
    - **Lagged and Moving Average Features:** To incorporate temporal dependencies, the model includes:
        - **Lagged Prices:** Closing prices from the previous three days (`lag_1`, `lag_2`, `lag_3`) were included as features, based on the principle of autocorrelation in financial time series.
        - **Moving Averages:** 7-day and 30-day simple moving averages (`ma_7`, `ma_30`) were calculated to smooth out short-term price fluctuations and identify longer-term trends.

### 2. Model Selection and Rationale

A diverse set of models was chosen to compare different learning paradigms:

- **Linear Models:**
    - **Linear Regression:** Chosen as a baseline model to establish a performance benchmark. It models a linear relationship between the engineered features and the target price.
    - **Ridge Regression:** Selected to address potential multicollinearity among the features (e.g., between moving averages and lagged prices). By applying L2 regularization, it helps prevent overfitting.
- **Non-linear Models:**
    - **Random Forest Regressor:** An ensemble method chosen for its high performance and ability to capture complex, non-linear relationships between features without requiring extensive hyperparameter tuning.
    - **Support Vector Machine (SVR):** Implemented to model the data in a high-dimensional feature space, making it effective at finding non-linear patterns.
- **Deep Learning Model:**
    - **Long Short-Term Memory (LSTM):** A recurrent neural network (RNN) architecture specifically designed for sequence and time-series data. It was chosen for its ability to learn long-term dependencies, which is highly relevant for financial forecasting. The model was structured with two LSTM layers and trained on scaled data to ensure numerical stability.

### 3. Training and Evaluation

- **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets chronologically (`shuffle=False`). This ensures that the models are trained on past data and evaluated on future data, simulating a real-world forecasting scenario.
- **Evaluation Metric:** **Mean Absolute Error (MAE)** was selected as the primary performance metric. MAE is intuitive, easy to interpret in the context of the exchange rate (e.g., an MAE of 0.5 means the model's predictions are, on average, off by 0.5 JPY), and is less sensitive to large, one-off prediction errors than Mean Squared Error (MSE).

## Quantitative Results

The models were trained and evaluated, yielding the following performance metrics. The results clearly indicate that simpler, linear models outperformed more complex approaches for this particular dataset and feature set.

| Model                    | Mean Absolute Error (MAE) |
| ------------------------ | ------------------------- |
| **Linear Regression**    | **0.7214**                |
| Ridge Regression         | 0.7421                    |
| Support Vector Machine   | 12.6945                   |
| LSTM                     | 12.3948                   |
| Random Forest            | 19.2807                   |

**Conclusion:** The **Linear Regression** model was identified as the best-performing model with an MAE of **0.7214**. This result suggests that, with the current feature engineering, a linear combination of the features provides the most accurate forecast. The higher MAE values for the non-linear and deep learning models suggest they may be overfitting to the training data or that the underlying relationships are predominantly linear.

Based on this analysis, the Linear Regression model was used to predict the next day's closing price.

**Prediction for the Next Trading Day:** **148.3744**

## Key Quantitative Skills Demonstrated

- **Time-Series Analysis:** Proficient in handling, cleaning, and transforming time-series data.
- **Feature Engineering:** Created and selected features (technical indicators, lags, moving averages) to improve model accuracy.
- **Statistical Modeling:** Implemented and evaluated a range of regression models, including Linear Regression, Ridge Regression, SVR, and Random Forest.
- **Deep Learning:** Built, trained, and evaluated an LSTM neural network for a forecasting task using TensorFlow/Keras.
- **Model Validation:** Employed a rigorous train-test split methodology suitable for time-series data and used MAE for quantitative model comparison.
- **Quantitative Reasoning:** Analyzed model performance metrics to draw data-driven conclusions and selected the optimal model for the final prediction.
- **Programming & Libraries:** Utilized Python with libraries such as `pandas` for data manipulation, `scikit-learn` for classical machine learning, `tensorflow/keras` for deep learning, and `ta` for financial indicator calculation.

## Getting Started

### Prerequisites

- Python 3.x
- An Alpha Vantage API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kotashida/usdjpy_predictor.git
    cd usdjpy_predictor
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    - Create a `.env` file in the root directory.
    - Add your Alpha Vantage API key to the `.env` file:
        ```
        ALPHA_VANTAGE_API_KEY=your_api_key
        ```

## Usage

To run the full analysis and generate a new prediction, execute the `main.py` script:

```bash
python src/main.py
```

The script will automatically fetch the latest data, process it, train and evaluate all models, and print a summary of the results before making a final prediction.