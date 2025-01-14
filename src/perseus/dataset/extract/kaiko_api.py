import ast
import json
import logging
import os
from os import path
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import requests
from perseus.settings import PROJECT_ROOT

KAIKO_API_KEY = os.environ.get("KAIKO_API_KEY")
if KAIKO_API_KEY:
    KAIKO_API_KEY = KAIKO_API_KEY.strip()
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


def get_kaiko_market_data(signal, ohlcv=False) -> pd.DataFrame:
    time.sleep(0.3)  # Add a delay to avoid hitting the rate limit
    # Parse the source_posted_at timestamp - sample : 2023-03-20 06:05:24+00:00
    source_posted_at = datetime.strptime(
        str(signal["source_posted_at"]), "%Y-%m-%d %H:%M:%S%z"
    )
    # Calculate the start_time and end_time for the data retrieval
    start_time = (source_posted_at - timedelta(days=0.5)).replace(
        tzinfo=None
    ).isoformat() + "Z"
    # end_time = source_posted_at.replace(tzinfo=None).isoformat() + "Z"
    end_time = (source_posted_at + timedelta(days=0.5)).replace(
        tzinfo=None
    ).isoformat() + "Z"

    try:
        # Make exchanges a list
        signal["exchanges"] = ast.literal_eval(str(signal["exchanges"]))
        # Extract the exchange code from the signal
        exchange_code = extract_exchange_code(signal["exchanges"][0])
    except:
        signal["exchanges"] = "Binance V2"
        exchange_code = extract_exchange_code(signal["exchanges"])
    # Set up the headers with your API key
    headers = {
        "Accept": "application/json",
        "X-Api-Key": str(KAIKO_API_KEY),
    }
    if ohlcv:
        # Build the complete URL with query parameters for OHLCV data
        url = f"https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/{exchange_code}/spot/{signal['commodity'].lower()}-{signal['base_commodity'].lower()}/aggregations/count_ohlcv_vwap"
        interval = "1m"  # You can modify the interval as needed
        url += f"?interval={interval}&start_time={start_time}&end_time={end_time}&page_size=5000"  # You can add more parameters if needed
        try:
            # Send the GET request to the Kaiko API
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception if the request is not successful

            # Parse the JSON response
            data = response.json()

            # Extract the OHLCV data from the response
            ohlcv_data = data.get("data", [])
            ohlcv_df = pd.DataFrame(ohlcv_data)
            # Print amount of data extracted
            print(
                f"Amount of OHLCV data extracted: {len(ohlcv_df)} on {signal['commodity']}-{signal['base_commodity']}"
            )
            return ohlcv_df
        except requests.exceptions.RequestException as e:
            print(f"Error on url: {url}")
            print(f"Error fetching Kaiko data: {str(e)}")
            return pd.DataFrame()

    else:
        try:
            # Initialize an empty list to hold trade data
            trade_data_list = []
            # Extract the trade data from the response
            continuation_token = None
            while True:
                if not signal["commodity"] or not signal["base_commodity"]:
                    break

                url = f"https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/{exchange_code}/spot/{signal['commodity'].lower()}-{signal['base_commodity'].lower()}/trades"

                if not continuation_token:
                    url = f"https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/{exchange_code}/spot/{signal['commodity'].lower()}-{signal['base_commodity'].lower()}/trades"
                    url += f"?start_time={start_time}&end_time={end_time}&page_size=100000"  # You can add more parameters if needed
                else:
                    url += f"?continuation_token={continuation_token}"

                print(url)
                response = requests.get(url, headers=headers)
                response_json = response.json()

                # Check for continuation token for the next iteration
                continuation_token = response_json.get("continuation_token")

                # Append fetched data to trade_data_list
                if "data" in response_json and isinstance(response_json["data"], list):
                    print(
                        f"Amount of trade data extracted: {len(response_json['data'])} on {signal['commodity']}-{signal['base_commodity']}"
                    )
                    trade_data_list.extend(response_json["data"])
                else:
                    print("The 'data' key is missing or is not a list.")

                if not continuation_token:
                    break

            # Convert the list of trade data into a DataFrame
            trade_df = pd.DataFrame(trade_data_list)

            return trade_df

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching Kaiko data: {str(e)}")
            return pd.DataFrame()


def plot_candlestick_and_volume(
    df,
    signal_time,
    annotation_text_size=16,
    axis_title_size=14,
    legend_text_size=12,
    tick_text_size=12,
    filename=path.join(PROJECT_ROOT, "data", "candlestick_chart.pdf"),
):
    """
    Plots a candlestick chart with volume bars and a signal time annotation.

    Args:
    df (DataFrame): A DataFrame containing 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    signal_time (datetime): The time of the signal to annotate on the chart.
    annotation_text_size (int): Font size for the signal time annotation.
    axis_title_size (int): Font size for the axis titles.
    legend_text_size (int): Font size for the legend text.
    tick_text_size (int): Font size for the tick text on the axes.
    """

    # Create the figure for the candlestick chart
    fig = go.Figure()

    # Add the candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candlestick",
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )

    # Add the volume trace as a bar chart using a single color
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            marker_color="blue",
            name="Volume",
            yaxis="y2",
            opacity=0.2,
        )
    )

    # Add annotation for the signal time
    fig.add_annotation(
        x=signal_time,
        y=0.75,
        yref="paper",
        text="Crowd Pump Announce Time",
        showarrow=True,
        arrowhead=1,
        arrowsize=2,
        arrowwidth=2,
        arrowcolor="Red",
        ax=0,
        ay=-60,
        font=dict(size=annotation_text_size),
    )

    # Update the layout to include a second y-axis for the volume, remove the rangeslider, and adjust the legend and tick text sizes
    fig.update_layout(
        xaxis_title="Date",
        xaxis=dict(
            rangeslider=dict(visible=False),
            title_font=dict(size=axis_title_size),
            tickfont=dict(size=tick_text_size),  # Set x-axis tick text size
        ),
        yaxis=dict(
            title="Price",
            side="left",
            title_font=dict(size=axis_title_size),
            tickfont=dict(size=tick_text_size),  # Set y-axis tick text size
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            title_font=dict(size=axis_title_size),
            tickfont=dict(size=tick_text_size),  # Set y-axis2 tick text size
        ),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.92,
            xanchor="right",
            x=1,
            font=dict(size=legend_text_size),
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    # Display the figure
    fig.show()

    # Save the figure to a PDF file if filename is provided
    if filename:
        pio.write_image(fig, filename, format="pdf")


# def plot_candlestick_and_volume(df, signal_time, annotation_text_size=16, axis_title_size=14, legend_text_size=12):
#     """
#     Plots a candlestick chart with volume bars and a signal time annotation.

#     Args:
#     df (DataFrame): A DataFrame containing 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
#     signal_time (datetime): The time of the signal to annotate on the chart.
#     annotation_text_size (int): Font size for the signal time annotation.
#     axis_title_size (int): Font size for the axis titles.
#     legend_text_size (int): Font size for the legend text.
#     """

#     # Create the figure for the candlestick chart
#     fig = go.Figure()

#     # Add the candlestick trace
#     fig.add_trace(
#         go.Candlestick(
#             x=df["timestamp"],
#             open=df["open"],
#             high=df["high"],
#             low=df["low"],
#             close=df["close"],
#             name="Candlestick",
#             increasing_line_color="green",
#             decreasing_line_color="red",
#         )
#     )

#     # Add the volume trace as a bar chart using a single color
#     fig.add_trace(
#         go.Bar(
#             x=df["timestamp"],
#             y=df["volume"],
#             marker_color="blue",  # Using blue for all volume bars
#             name="Volume",
#             yaxis="y2",  # Indicates this trace uses the secondary y-axis
#             opacity=0.2,  # Adjust the opacity here; for example, 0.5 for semi-transparent
#         )
#     )

#     # Add annotation as a pseudo-legend for the signal time
#     fig.add_annotation(
#         x=signal_time,
#         y=0.75,
#         yref="paper",
#         text="Crowd Pump Announce Time",
#         showarrow=True,
#         arrowhead=1,
#         arrowsize=2,
#         arrowwidth=2,
#         arrowcolor="Red",
#         ax=0,  # Horizontal end point of the arrow (0 means no horizontal shift)
#         ay=-60,
#         font=dict(size=annotation_text_size),  # Adjust font size as per parameter
#     )

#     # Update the layout to include a second y-axis for the volume, remove the rangeslider, and adjust the legend
#     fig.update_layout(
#         xaxis_title="Date",
#         xaxis=dict(
#             rangeslider=dict(
#                 visible=False  # Disable the rangeslider to remove the scrollbar at the bottom
#             ),
#             title_font=dict(size=axis_title_size)  # Set axis title font size
#         ),
#         yaxis=dict(title="Price", side="left", title_font=dict(size=axis_title_size)),  # Price axis on the left side
#         yaxis2=dict(
#             title="Volume",
#             overlaying="y",  # Share the same x-axis
#             side="right",  # Volume axis on the right side
#             showgrid=False,  # Hide gridlines for volume axis for clarity
#             title_font=dict(size=axis_title_size)  # Set axis title font size
#         ),
#         legend=dict(
#             orientation="v",  # Make legend vertical
#             yanchor="middle",
#             y=0.92,
#             xanchor="right",
#             x=1,  # Place the legend outside the plot to the right
#             font=dict(size=legend_text_size),  # Adjust legend font size
#             bgcolor="rgba(255,255,255,0)",  # Set background to transparent
#         ),
#     )

#     # Display the figure
#     fig.show()


if __name__ == "__main__":

    signal = pd.Series(
        {
            "source_posted_at": "2024-05-01 02:22:36+00:00",
            "commodity": "VET",
            "base_commodity": "USDT",
            "exchanges": "[]",
        }
    )

    # Extracting the source_posted_at time from the signal
    signal_time = datetime.strptime(
        signal["source_posted_at"], "%Y-%m-%d %H:%M:%S%z"
    ).replace(tzinfo=None)

    # Example: Retrieve OHLCV data
    ohlcv_df = get_kaiko_market_data(signal, ohlcv=True)

    if ohlcv_df is not None:
        print(ohlcv_df.head(20))
        print(ohlcv_df.shape)
        print(datetime.fromtimestamp(ohlcv_df["timestamp"].iloc[0] / 1000))
        print(datetime.fromtimestamp(ohlcv_df["timestamp"].iloc[-1] / 1000))
        # Plot the OHLCV data

    # Example: Retrieve trade data
    trade_df = get_kaiko_market_data(signal, ohlcv=False)

    if trade_df is not None:
        print(trade_df.head(20))
        print(trade_df.shape)
        print(datetime.fromtimestamp(trade_df["timestamp"].iloc[0] / 1000))
        print(datetime.fromtimestamp(trade_df["timestamp"].iloc[-1] / 1000))
        # Plot the trade data

    # Creating a DataFrame
    df = ohlcv_df

    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x / 1000))

    # Convert open, high, low, close, volume, and price columns to numeric values
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df.set_index("timestamp", inplace=True)

    df.dropna(inplace=True)

    # Resampling data into hourly format
    hourly_df = df.resample("30T").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "price": "mean",
            "count": "sum",
        }
    )

    hourly_df = hourly_df.reset_index()

    # Example of usage
    # Assuming df and signal_time are already defined as per the user's data.
    plot_candlestick_and_volume(
        hourly_df,
        signal_time,
        annotation_text_size=18,
        axis_title_size=18,
        legend_text_size=18,
        tick_text_size=18,
    )

    # # Create the figure for the candlestick chart
    # fig = go.Figure()

    # # Add the candlestick trace
    # fig.add_trace(
    #     go.Candlestick(
    #         x=hourly_df["timestamp"],
    #         open=hourly_df["open"],
    #         high=hourly_df["high"],
    #         low=hourly_df["low"],
    #         close=hourly_df["close"],
    #         name="VET Candlestick",
    #         increasing_line_color="green",
    #         decreasing_line_color="red",
    #     )
    # )

    # # Add the volume trace as a bar chart using a single color
    # fig.add_trace(
    #     go.Bar(
    #         x=hourly_df["timestamp"],
    #         y=hourly_df["volume"],
    #         marker_color="blue",  # Using blue for all volume bars
    #         name="Volume",
    #         yaxis="y2",  # Indicates this trace uses the secondary y-axis
    #         opacity=0.2,  # Adjust the opacity here; for example, 0.5 for semi-transparent
    #     )
    # )

    # # Extracting the source_posted_at time from the signal
    # signal_time = datetime.strptime(
    #     signal["source_posted_at"], "%Y-%m-%d %H:%M:%S%z"
    # ).replace(tzinfo=None)

    # # # Add the vertical line for signal time on the plot and make it bolder
    # # fig.add_shape(type="line",
    # #             x0=signal_time, x1=signal_time,
    # #             y0=0, y1=1, yref="paper",
    # #             line=dict(color="Red", width=4))

    # # Add annotation as a pseudo-legend for the signal time
    # fig.add_annotation(
    #     x=signal_time,
    #     y=0.75,
    #     yref="paper",
    #     text="Crowd Pump Announce Time",
    #     showarrow=True,
    #     arrowhead=1,
    #     arrowsize=2,
    #     arrowwidth=2,
    #     arrowcolor="Red",
    #     ax=0,  # Horizontal end point of the arrow (0 means no horizontal shift)
    #     ay=-60,
    #     font=dict(size=16),  # Change size to your preference
    # )

    # # Update the layout to include a second y-axis for the volume, remove the rangeslider, and adjust the legend
    # fig.update_layout(
    #     # title="BTC Market Data and Volume",
    #     xaxis_title="Date",
    #     xaxis=dict(
    #         rangeslider=dict(
    #             visible=False  # Disable the rangeslider to remove the scrollbar at the bottom
    #         )
    #     ),
    #     yaxis=dict(title="Price", side="left"),  # Price axis on the left side
    #     yaxis2=dict(
    #         title="Volume",
    #         overlaying="y",  # Share the same x-axis
    #         side="right",  # Volume axis on the right side
    #         showgrid=False,  # Hide gridlines for volume axis for clarity
    #     ),
    #     legend=dict(
    #         orientation="v",  # Make legend vertical
    #         yanchor="middle",
    #         y=0.92,
    #         xanchor="right",
    #         x=1,  # Place the legend outside the plot to the right
    #         bgcolor="rgba(255,255,255,0)",  # Set background to transparent
    #     ),
    # )

    # # Display the figure
    # fig.show()
