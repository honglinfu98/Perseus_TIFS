"""
Module: t2_display_summary_statistics.py
----------------------------------------
Display the summary statistics CSV saved by t2_summary_statistics.py.
Run it as a script or in the interactive window (Shift+Enter on the cell).
"""

# %%
from os import path

import pandas as pd

from perseus.settings import PROJECT_ROOT

csv_path = path.join(PROJECT_ROOT, "results", "t2_summary_statistics.csv")

df = pd.read_csv(csv_path, index_col="statistic")
df["total"] = df.sum(axis=1)
print(df)
df
