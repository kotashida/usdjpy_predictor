
import os
import requests
import pandas as pd
from dotenv import load_dotenv

def fetch_usdjpy_data():
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("Error: ALPHA_VANTAGE_API_KEY not found in .env file.")
        return

    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=JPY&apikey={api_key}&outputsize=full"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "Time Series FX (Daily)" not in data:
            print("Error: Could not fetch data.")
            print("Response from server:", data)
            return

        df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        })
        df = df.apply(pd.to_numeric)
        df = df.sort_index()

        file_path = "usdjpy_daily.csv"
        df.to_csv(file_path, index_label="date")
        print(f"Successfully saved data to {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage: {e}")

if __name__ == "__main__":
    fetch_usdjpy_data()
