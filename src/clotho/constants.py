"""
Constants used for this project 
"""
from os import path
from clotho.settings import PROJECT_ROOT

FIGURES_PATH = path.join(PROJECT_ROOT, "figures")
DATA_PATH = path.join(PROJECT_ROOT, "data")


class DataSchema:
    """
    Class for storing the ids for html and dash
    """

    COMMODITY = "commodity"
    USERNAME = "username"
    SOURCE_DATETIME = "source_datetime"
    MONTH = "month"
    YEAR = "year"
    TYPE = "signal_type"
    DATE = "date"
    SIZE = "size"
    FREQ = "freq"
    PID = "pid"
    ENTITY_ID = "entity_id"


DATE_PICKER_RANGE_CONTAINER = "output-container-date-picker-range"
DATE_PICKER_RANGE = "date-picker-range"

CHANNELS_NETWORK_CHART = "channel-network-chart"
COMMODITY_SCATTER_CHART = "commodity-scatter-chart"
CHANNEL_SCATTER_CHART = "channel-scatter-chart"
BUBBLE_CHART = "bubble-chart"
TYPE_PIE_CHART = "type-pie-chart"
COMMODITY_BAR_CHART = "commodity-bar-chart"
CHANNEL_BAR_CAHRT = "channel-bar-chart"
SUMMARY_TEXT = "summary_text"
HOUR_CHART = "hour_chart"

SELECT_ALL_TYPE_BUTTON = "select-all-type-button"
TYPE_DROPDOWN = "type-dropdown"

SELECT_ALL_CHANNEL_BUTTON = "select-all-channel-button"
CHANNEL_DROPDOWN = "channel-dropdown"

SELECT_ALL_COMMODITY_BUTTON = "select-all-commodity-button"
COMMODITY_DROPDOWN = "commodity-dropdown"

SELECT_ALL_MONTHS_BUTTON = "select-all-months-button"
MONTH_DROPDOWN = "month-dropdown"

SELECT_ALL_YEARS_BUTTON = "select-all-years-button"
YEAR_DROPDOWN = "year-dropdown"


# Ignore list
IGNORED_COMMODITY = [
    "000LUNA",
    "BUSD",
    "G",
    "T",
    "ENTRY",
    "DAY",
    "D",
    "RICH",
    "L",
    "SWING",
    "",
    "TRADE",
    "LONG",
    "AT",
    "LESS",
    "NEXT",
    "1500",
    "BINANCE",
    "42",
    "365",
    "120B",
    "1717",
    "1150",
    "STOP",
    "ETH",
    "BTC",
    "112",
    "21200",
    "MEGA",
    "FOR",
    "500",
    "USDT",
    "ON",
    "TOKEN",
    "LET",
    "GO",
    "300",
    "TODAY",
    "NEW",
    "COIN",
    "BUY",
    "NOW",
    "GMT",
    "VIP",
    "00",
    "42",
    "OUR",
    "POST",
    "GET",
    "NAME",
    "COM",
    "ERC",
    "SMART",
    "HUGE",
]
