import pandas as pd
import plotly.express as px
from perseus.dataset.preprocess.train_test_validate import get_test_scored_signals, get_train_scored_signals, get_valid_scored_signals
from perseus.dataset.preprocess.process import (
    aggregate_data,
    assign_event_ids,
    process_dataframe,
    get_graphs,
)


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
    signals = get_test_scored_signals()
    processed_signals = process_dataframe(signals)
    ided_signals = assign_event_ids(processed_signals)
    cascade, no_nodes, id_mapping, labeling_oldid = aggregate_data(ided_signals)
    gs, results, As, P_dict = get_graphs(cascade, no_nodes, id_mapping)
    filtered_signals = processed_signals[
        processed_signals["commodity"].isin([key for key, value in gs.items()])
    ]
    plot_paper_with_plotly_alphabet_fixed(filtered_signals)
