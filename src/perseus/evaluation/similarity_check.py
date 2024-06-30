from perseus.dataset.preprocess.train_test_validate import get_test_scored_signals, get_train_scored_signals, get_valid_scored_signals
from perseus.dataset.preprocess.process import process_dataframe
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from sklearn.cluster import DBSCAN



signals = get_train_scored_signals()
processed_signals = process_dataframe(signals)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')



def get_embeddings(texts):
    # Tokenize and encode the batch of texts
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encoded_input)
    # Extract the embeddings of the [CLS] token for each text
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()



cluster_results = {}



# Modify the clustering part in the loop
for commodity, group in processed_signals.groupby('commodity'):
    texts = group['message_text'].tolist()
    embeddings = get_embeddings(texts)

    # Use DBSCAN instead of HDBSCAN
    clusterer = DBSCAN(eps=0.1, min_samples=2)  # Adjust eps and min_samples as needed
    cluster_labels = clusterer.fit_predict(embeddings)

    commodity_results = group.copy()
    commodity_results['cluster'] = cluster_labels
    cluster_results[commodity] = commodity_results


