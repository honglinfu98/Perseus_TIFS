"""
Module: t3_display_multi_results.py
-----------------------------------
Display the model comparison table saved by t3_multi_results.py.
Run it as a script or in the interactive window (Shift+Enter on the cell).
"""

# %%
from os import path

import pandas as pd

from perseus.settings import PROJECT_ROOT

csv_path = path.join(PROJECT_ROOT, "results", "t3_multi_results.csv")

df = pd.read_csv(csv_path)
df = df.set_index("model_dataset")
df_display = df.round(3)
print(df_display)
df_display
