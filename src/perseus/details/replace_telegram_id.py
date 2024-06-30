from os import path
import pandas as pd
from perseus.dataset.preprocess.process import aggregate_data, assign_event_ids, process_dataframe
from perseus.dataset.preprocess.train_test_validate import get_train_scored_signals
from perseus.settings import PROJECT_ROOT



signals = get_train_scored_signals()
processed_signals = process_dataframe(signals)
ided_signals = assign_event_ids(processed_signals)
cascade, no_nodes, id_mapping, cascade_labeling = aggregate_data(ided_signals)





a = pd.read_csv(path.join(PROJECT_ROOT, "data", "telegram_id.csv"))
id_to_username = dict(zip(a.telegram_chat_id, a.username))

# Replacing the first element of each tuple in each sublist with the corresponding username
cascade_labeling["POWR"] = [
    [tuple([id_to_username[t[0]] if t[0] in id_to_username else t[0]] + list(t[1:])) for t in sublist]
    for sublist in cascade_labeling["POWR"]
]