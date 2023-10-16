import matplotlib.pyplot as plt
import pandas as pd
from clotho.extract.cloudburst_connection import get_scored_signals
from clotho.post_processing.graph_inferring import get_graphs
from clotho.pre_processing_summary.scored_signals import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    features_engineer,
)
import plotly.express as px
import plotly.graph_objects as go


# def plot_paper(processed_signals: pd.DataFrame):
#     # unique_commodities = [key for key, value in scores.items()]
#     # Identify unique commodities
#     unique_commodities = processed_signals["commodity"].unique()

#     # Create plots for each commodity
#     # plots = len(unique_commodities)
#     for commodity in unique_commodities:
#         subset_data = processed_signals[processed_signals["commodity"] == commodity]

#         plt.figure(figsize=(15, 10))
#         plt.scatter(
#             subset_data["start_date"],
#             subset_data["increase_percentage"],
#             color="blue",
#             marker="o",
#         )
#         for i, txt in enumerate(subset_data["telegram_chat_id"]):
#             plt.annotate(
#                 txt,
#                 (
#                     subset_data["start_date"].iloc[i],
#                     subset_data["increase_percentage"].iloc[i],
#                 ),
#                 fontsize=9,
#             )

#         plt.title(f"Increase Percentage over Time for {commodity}")
#         plt.xlabel("Date")
#         plt.ylabel("Increase Percentage")
#         plt.xticks(rotation=45)
#         plt.grid(True, which="both", ls="--", c="0.65")
#         plt.tight_layout()
#         plt.show()

# def plot_paper_with_plotly_alphabet_direct(processed_signals: pd.DataFrame):
#     # Identify unique commodities
#     unique_commodities = processed_signals["commodity"].unique()

#     # Create plots for each commodity
#     for commodity in unique_commodities:
#         subset_data = processed_signals[processed_signals["commodity"] == commodity]

#         # Create hover text with wrapped message_text and telegram_chat_id
#         subset_data['hover_text'] = "ID: " + subset_data['telegram_chat_id'].astype(str) + "<br>" + subset_data['message_text'].str.wrap(30).str.replace("\n", "<br>")

#         fig = px.scatter(subset_data,
#                          x='start_date',
#                          y='increase_percentage',
#                          color='telegram_chat_id',  # Color by telegram_chat_id
#                          hover_data=['hover_text'],  # Display wrapped message_text and telegram_chat_id on hover
#                          color_discrete_sequence=distinct_palette_alphabet,  # Use Alphabet palette directly
#                          title=f"Increase Percentage over Time for {commodity}",
#                          labels={"start_date": "Date", "increase_percentage": "Increase Percentage"})

#         # Adjust the plot size
#         fig.update_layout(height=600, width=1000)

#         fig.show()


# def plot_paper_with_plotly_alphabet_fixed(processed_signals: pd.DataFrame):
#     # Define the Alphabet palette directly
#     distinct_palette_alphabet_fixed = px.colors.qualitative.Alphabet

#     # Identify unique commodities
#     unique_commodities = processed_signals["commodity"].unique()

#     # Create a mapping for position to symbols
#     symbol_mapping = {
#         'long': 'circle',
#         'short': 'cross'
#     }
#     processed_signals['symbol'] = processed_signals['position'].map(symbol_mapping)

#     # Create plots for each commodity
#     for commodity in unique_commodities:
#         subset_data = processed_signals[processed_signals["commodity"] == commodity]
#         subset_data["telegram_chat_id"] = subset_data["telegram_chat_id"].astype('category')

#         # Create hover text with wrapped message_text and telegram_chat_id
#         subset_data['hover_text'] = "ID: " + subset_data['telegram_chat_id'].astype(str) + "<br>" + subset_data['message_text'].str.wrap(30).str.replace("\n", "<br>")

#         fig = px.scatter(subset_data,
#                          x='start_date',
#                          y='increase_percentage',
#                          color='telegram_chat_id',  # Color by telegram_chat_id
#                          hover_data=['hover_text'],  # Display wrapped message_text and telegram_chat_id on hover
#                          color_discrete_sequence=distinct_palette_alphabet_fixed,  # Use Alphabet palette directly
#                          symbol='symbol',  # Use symbols based on the 'position'
#                          title=f"Increase Percentage over Time for {commodity}",
#                          labels={"start_date": "Date", "increase_percentage": "Increase Percentage"})

#         # Adjust the plot size
#         fig.update_layout(height=600, width=1000)

#         fig.show()


# def plot_paper_with_plotly_alphabet_fixed(processed_signals: pd.DataFrame):
#     # Define the Alphabet palette directly
#     distinct_palette_alphabet_fixed = px.colors.qualitative.Alphabet

#     # Identify unique commodities
#     unique_commodities = processed_signals["commodity"].unique()

#     # Create plots for each commodity
#     for commodity in unique_commodities:
#         subset_data = processed_signals[processed_signals["commodity"] == commodity]
#         subset_data["telegram_chat_id"] = subset_data["telegram_chat_id"].astype('category')

#         # Create hover text with wrapped message_text and telegram_chat_id
#         subset_data['hover_text'] = "ID: " + subset_data['telegram_chat_id'].astype(str) + "<br>" + subset_data['message_text'].str.wrap(30).str.replace("\n", "<br>")

#         fig = px.scatter(subset_data,
#                          x='start_date',
#                          y='increase_percentage',
#                          color='telegram_chat_id',  # Color by telegram_chat_id
#                          hover_data=['hover_text'],  # Display wrapped message_text and telegram_chat_id on hover
#                          color_discrete_sequence=distinct_palette_alphabet_fixed,  # Use Alphabet palette directly
#                          title=f"Increase Percentage over Time for {commodity}",
#                          labels={"start_date": "Date", "increase_percentage": "Increase Percentage"})

#         # Adjust the plot size
#         fig.update_layout(height=600, width=1000)

#         fig.show()

# Note: As the data isn't loaded currently, the function will not be tested here.
# You can use this corrected function in your local environment with the 'filtered_signals' data.
import pandas as pd
import plotly.express as px


def plot_paper_with_plotly_alphabet_fixed(processed_signals: pd.DataFrame):
    # Define the Alphabet palette directly
    distinct_palette_alphabet_fixed = px.colors.qualitative.Alphabet

    # Identify unique commodities
    unique_commodities = processed_signals["commodity"].unique()

    # Create a mapping for position to symbols
    symbol_mapping = {"long": "circle", "short": "cross"}
    processed_signals["symbol"] = processed_signals["position"].map(symbol_mapping)

    # Create plots for each commodity
    for commodity in unique_commodities:
        subset_data = processed_signals[processed_signals["commodity"] == commodity]
        subset_data["telegram_chat_id"] = subset_data["telegram_chat_id"].astype(
            "category"
        )

        # Create hover text with wrapped message_text and telegram_chat_id
        subset_data["hover_text"] = (
            "ID: "
            + subset_data["telegram_chat_id"].astype(str)
            + "<br>"
            + subset_data["message_text"].str.wrap(30).str.replace("\n", "<br>")
        )

        fig = px.scatter(
            subset_data,
            x="start_date",
            y="increase_percentage",
            color="telegram_chat_id",  # Color by telegram_chat_id
            hover_data=[
                "hover_text"
            ],  # Display wrapped message_text and telegram_chat_id on hover
            color_discrete_sequence=distinct_palette_alphabet_fixed,  # Use Alphabet palette directly
            symbol="symbol",  # Use symbols based on the 'position'
            title=f"Increase Percentage over Time for {commodity}",
            labels={"start_date": "Date", "increase_percentage": "Increase Percentage"},
        )

        # Adjust the plot size
        fig.update_layout(height=600, width=1000)

        fig.show()


if __name__ == "__main__":
    signals = get_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping = aggregate_data(ided_signals)
    gs = get_graphs(cascade, no_nodes, id_mapping)
    # features = features_engineer(processed_signals)

    filtered_signals = processed_signals[
        processed_signals["commodity"].isin([key for key, value in gs.items()])
    ]

    plot_paper_with_plotly_alphabet_fixed(filtered_signals)
