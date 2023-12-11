from clotho.extract.cloudburst_connection import get_masterminds


# Function to create label mapping based on top n frequency
def create_label_mapping(commodity_frequency_lists, n):
    label_mapping = {}
    for commodity, freq_list in commodity_frequency_lists.items():
        # Sort and take top n
        top_n_chat_ids = set(
            [
                chat_id
                for chat_id, _ in sorted(freq_list, key=lambda x: x[1], reverse=True)[
                    :n
                ]
            ]
        )
        label_mapping[commodity] = {
            chat_id: 1 if chat_id in top_n_chat_ids else 0 for chat_id, _ in freq_list
        }
    return label_mapping


if __name__ == "__main__":
    # signals = get_scored_signals()
    data = get_masterminds()

    # Grouping by commodity and telegram_chat_id and counting the occurrences
    commodity_frequency = (
        data.groupby(["commodity", "telegram_chat_id"])
        .size()
        .reset_index(name="frequency")
    )

    # Sorting within each commodity group by frequency in descending order
    commodity_frequency_sorted = commodity_frequency.sort_values(
        ["commodity", "frequency"], ascending=[True, False]
    )

    # Extracting the sorted list for each commodity
    commodity_frequency_lists = commodity_frequency_sorted.groupby("commodity").apply(
        lambda x: x[["telegram_chat_id", "frequency"]].values.tolist()
    )

    # Example: Create label mapping for top 3 telegram_chat_ids for each commodity
    n = 3
    label_mapping = create_label_mapping(commodity_frequency_lists, n)
