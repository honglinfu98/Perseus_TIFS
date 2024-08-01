"""
Used to get the circulation of all the coins and save it as a dictionary
"""

from os import path
import pickle
import coinmarketcapapi
from perseus.settings import PROJECT_ROOT
from perseus.config import COINMARKETCAP_API_KEY


def get_coin_circulation():
    """
    get the market cap of all the coins and save it as a dictionary
    """
    api_key_parameter = COINMARKETCAP_API_KEY
    cmc = coinmarketcapapi.CoinMarketCapAPI(api_key_parameter)
    data_listing = cmc.cryptocurrency_listings_latest(limit=5000)
    keys = [i["symbol"] for i in data_listing.data]
    values = [i["circulating_supply"] for i in data_listing.data]
    coin_circulating_buffer = dict(zip(keys, values))
    self_circulating = [
        i["self_reported_circulating_supply"] for i in data_listing.data
    ]
    coin_cap_selfreported = dict(zip(keys, self_circulating))

    for k, v in coin_circulating_buffer.items():
        if v == 0:
            coin_circulating_buffer[k] = coin_cap_selfreported[k]

    return coin_circulating_buffer


if __name__ == "__main__":
    coin_circulating = get_coin_circulation()
    with open(path.join(PROJECT_ROOT, "data", "coin_circulating.pkl"), "wb") as f:
        pickle.dump(coin_circulating, f)
