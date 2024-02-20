from os import path
import pickle
from perseus.settings import PROJECT_ROOT

import pandas as pd

with open(path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","all_scored.pkl"), "rb") as file:
    df = pickle.load(file)        


df = df[~df["telegram_chat_id"].isna()]

# Assuming `df` is your DataFrame

# 1. Convert 'created_at' to datetime

# 2. Filter the dataset for the specified date range
start_date = '2023-11-01'
end_date = '2024-02-01'


train_df = df[(df['source_posted_at'] < start_date)]

test_df = df[(df['source_posted_at'] >= start_date) & (df['source_posted_at'] <= end_date)]

validate_df = df[(df['source_posted_at'] > end_date)]




with open(path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","train_signals.pkl"), "wb") as file:
    pickle.dump(train_df,file)        

with open(path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","test_signals.pkl"), "wb") as file:
    pickle.dump(test_df,file)        

with open(path.join(PROJECT_ROOT, "src","clotho","dataset","extract","download","validate_signals.pkl"), "wb") as file:
    pickle.dump(validate_df,file)        
