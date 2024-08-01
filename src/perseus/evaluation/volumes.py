"""
This script is used to compare the normal trading volumes with average trading volumes
"""

from os import path
import pandas as pd
from matplotlib import pyplot as plt
from perseus.dataset.extract.cloudburst_connection import get_volumes

volume = get_volumes()


# Calculate the normalized trading volume
minutes_in_3_days = 24 * 60 * 3
volume["normalized_volume"] = (volume["total_volume_bc"] / minutes_in_3_days) * volume[
    "duration_min"
]


# Aggregate the values
total_volume_traded = volume["volume_traded_bc"].sum()
total_normalized_volume = volume["normalized_volume"].sum()

# Create a DataFrame for the aggregated values
aggregated_data = {
    "Volume Type": ["Total Volume Traded", "Total Normalized Volume"],
    "Volume": [total_volume_traded, total_normalized_volume],
}
aggregated_df = pd.DataFrame(aggregated_data)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(aggregated_df["Volume Type"], aggregated_df["Volume"], color=["blue", "green"])
plt.title("Comparison of Total Volume Traded vs. Total Normalized Volume")
plt.xlabel("Volume Type")
plt.ylabel("Volume")
plt.show()
