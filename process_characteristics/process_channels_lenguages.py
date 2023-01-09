"""
This script is used to process the channels to extract the characteristics for later use
on profiling the channels
"""
import pandas as pd
#import numpy as np
from extraction.loader import load_cloudburst_signals

def call_cloudburst_signals()->pd.DataFrame:
    """
    Call the cloudburst signals
    output: pd.DataFrame
    """
    cloudbusrt_signals = load_cloudburst_signals()
    return cloudbusrt_signals

if __name__ == '__main__':
    cloudbusrt_signals_df = call_cloudburst_signals()
