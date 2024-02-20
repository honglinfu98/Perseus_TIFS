"""
This script is used to get the market cap of all the coins and save it as a dictionary
"""
import os
from os import path
import pickle
from settings import PROJECT_ROOT
import coinmarketcapapi


COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")


def get_coin_circulation():
    """
    get the market cap of all the coins and save it as a dictionary
    """
    api_key_parameter = COINMARKETCAP_API_KEY
    cmc = coinmarketcapapi.CoinMarketCapAPI(api_key_parameter)
    data_listing = cmc.cryptocurrency_listings_latest(limit=5000)
    keys = [i["symbol"] for i in data_listing.data]
    values = [i["circulating_supply"] for i in data_listing.data]
    coin_circulating = dict(zip(keys, values))
    self_circulating = [
        i["self_reported_circulating_supply"] for i in data_listing.data
    ]
    coin_cap_selfreported = dict(zip(keys, self_circulating))
    for coin in coin_circulating.keys():
        if coin_circulating[coin] == 0:
            coin_circulating[coin] = coin_cap_selfreported[coin]
    return coin_circulating


if __name__ == "__main__":
    coin_circulating = get_coin_circulation()
    with open(path.join(PROJECT_ROOT, "data", "coin_circulating.pkl"), "wb") as f:
        pickle.dump(coin_circulating, f)
