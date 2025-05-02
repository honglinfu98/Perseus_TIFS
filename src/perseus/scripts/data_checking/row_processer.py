import pandas as p
import ast
import json
import logging
import os
from os import path
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
from os import path
import numpy as np
from perseus.dataset.extract.cloudburst_connection import get_new_detection
from perseus.dataset.preprocess.train_test_validate import (
    get_test_scored_signals,
    get_train_scored_signals,
    get_valid_scored_signals,
)

from perseus.settings import PROJECT_ROOT
from concurrent.futures import ProcessPoolExecutor, as_completed

KAIKO_API_KEY = os.environ.get("KAIKO_API_KEY")


kaiko_exchanges_path = path.join(PROJECT_ROOT, "data/exchanges_kaiko.json")


def extract_exchange_code(name: str) -> str:
    """
    opens .json file with exchange codes and name and returns code for exchange
    don't care if the key 'name' is in upper or lower case
    :param exhange: str
    :return: str
    """
    exchange_codes = []
    with open(kaiko_exchanges_path, "r") as f:
        exchange_codes = json.load(f)
    for item in exchange_codes["data"]:
        if item["name"].lower() == name.lower():
            return item["code"]
    return f"Exchange {name} not found"


exchange_code = extract_exchange_code("Binance V2")


def process_row(row_dict):
    try:
        id = row_dict["id"]
        duration_info = json.loads(row_dict["duration"])
        start = datetime.strptime(duration_info["start_date"], "%Y-%m-%d %H:%M:%S.%f")
        end = datetime.strptime(duration_info["end_date"], "%Y-%m-%d %H:%M:%S.%f")

        start_time = start.replace(tzinfo=None).isoformat() + "Z"
        end_time = end.replace(tzinfo=None).isoformat() + "Z"

        exchange_code = extract_exchange_code("Binance V2")
        headers = {
            "Accept": "application/json",
            "X-Api-Key": str(KAIKO_API_KEY),  # Make sure this is global or passed
        }

        interval = "1m"
        url = (
            f"https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/{exchange_code}/spot/btc-usdt/aggregations/count_ohlcv_vwap"
            f"?interval={interval}&start_time={start_time}&end_time={end_time}&page_size=5000"
        )

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        ohlcv_data = data.get("data", [])
        ohlcv_df = pd.DataFrame(ohlcv_data)

        print(
            f"Row {id}: Amount of OHLCV data extracted: {len(ohlcv_data)} on btc-usdt"
        )

    except Exception as ex:
        print(f"Row {id}: Error fetching or processing OHLCV data: {str(ex)}")
        return None

    try:
        price_increase_info = json.loads(row_dict["price_increase"])
        if len(ohlcv_data) == 0:
            increase_percentage = price_increase_info["price_increase"]
        else:
            price = pd.to_numeric(ohlcv_df["price"])
            to_price = price_increase_info["to"]
            from_price = price_increase_info["from"]
            increase_percentage = (
                (price.iloc[-1] * to_price) - (price.iloc[0] * from_price)
            ) / (price.iloc[-1] * to_price)

        return {"id": id, "increase_percentage": increase_percentage}

    except Exception as ex:
        print(f"Row {id}: Error in price increase calculation: {str(ex)}")
        return None
