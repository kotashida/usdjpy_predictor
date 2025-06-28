# USD/JPY Exchange Rate Predictor

This project predicts the daily closing price of the USD/JPY exchange rate using various machine learning models. It fetches historical data from Alpha Vantage, performs feature engineering, and compares the performance of several models to identify the best one for prediction.

## Features

*   Fetches historical USD/JPY data using the Alpha Vantage API.
*   Performs feature engineering by adding technical indicators (RSI, MACD, Bollinger Bands) and lagged features.
*   Compares the performance of the following models:
    *   Linear Regression
    *   Ridge Regression
    *   Random Forest Regressor
    *   Support Vector Machine (SVM)
    *   Long Short-Term Memory (LSTM)
*   Identifies the best-performing model based on Mean Absolute Error (MAE).
*   Predicts the next day's closing price using the best model.

## Getting Started

### Prerequisites

*   Python 3.x
*   An Alpha Vantage API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/usdjpy_predictor.git
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
    *   Create a `.env` file in the root directory.
    *   Add your Alpha Vantage API key to the `.env` file:
        ```
        ALPHA_VANTAGE_API_KEY=your_api_key
        ```

## Usage

1.  **Fetch the latest data:**
    ```bash
    python data_fetcher.py
    ```
    This will download the latest USD/JPY data and save it as `usdjpy_daily.csv`.

2.  **Run the model comparison and prediction:**
    ```bash
    python main.py
    ```
    This will:
    *   Load the data from `usdjpy_daily.csv`.
    *   Train and evaluate all the models.
    *   Print a summary of the model performance.
    *   Predict the next day's closing price using the best-performing model.

## Model Comparison

The `main.py` script compares the following models:

*   **Traditional Models:**
    *   Linear Regression
    *   Ridge Regression
    *   Random Forest
    *   Support Vector Machine
*   **Deep Learning Model:**
    *   Long Short-Term Memory (LSTM)

The models are evaluated using Mean Absolute Error (MAE), and the best-performing model is used for the final prediction.
