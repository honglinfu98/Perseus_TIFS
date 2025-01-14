from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from perseus.settings import PROJECT_ROOT


with open(path.join(PROJECT_ROOT, "data", "comparison.pkl"), "rb") as file:
    comparison = pickle.load(file)

df = pd.DataFrame(comparison)


# Convert the dates from string to datetime
df["source_posted_at"] = pd.to_datetime(df["source_posted_at"])

# Extract year from datetime
df["year"] = df["source_posted_at"].dt.year

# Count the occurrences of each fraud type per year
pivot_df = df.pivot_table(
    index="year", columns="fraud_type", aggfunc="size", fill_value=0
)

# Calculate percentage for each fraud type per year
pivot_percentage = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100


# Increase the size of texts in the plot
plt.rcParams.update({"font.size": 12})  # Updates the default text sizes
filtered_df = pivot_percentage[pivot_percentage.index >= 2018]

# filtered_df = pivot_percentage[pivot_percentage.index >= 2018]
filtered_df.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color={"timepump": "lightblue", "crowdpump": "lightcoral"},
)
# plt.title('Percentage of Pump and Dump Cases from 2018 Onwards', fontsize=18)  # Larger title
plt.xlabel("Years", fontsize=18)  # Larger x-axis label
plt.ylabel("Percentage of Pump-and-dump Cases", fontsize=18)  # Larger y-axis label
plt.xticks(rotation=45, fontsize=18)  # Larger x-axis tick labels
plt.yticks(fontsize=18)  # Larger y-axis tick labels
plt.legend(
    title="Fraud Type",
    title_fontsize="18",
    fontsize="18",
    labels=["Crowd Pump", "Time Pump"],
    loc="lower right",
)  # Larger legend
plt.savefig(path.join(PROJECT_ROOT, "data", "comparison_plot.pdf"))
plt.show()


# save the plot to data folder as pdf
