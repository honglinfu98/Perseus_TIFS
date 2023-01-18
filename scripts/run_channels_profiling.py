"""
This script runs the flowork of the channels profiling.
"""

import logging
from clotho.extraction.loader import load_cloudburst_signals
from clotho.profiling.process_channels_lenguages import detect_and_group_by_language

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

# Function for exploring the lenguages of the channels
# First it's necessary conver langs_detected to a dataframe
# lenguages_channels.loc[lambda x: x[2].apply(lambda y: 'vi' in y)]

if __name__ == "__main__":
    logger.info("Loading cloudburst signals...")
    cloudbusrt_signals_df = load_cloudburst_signals()
    logger.info("Cloudburst signals loaded")
    logger.info("Detecting and grouping by channels language...")
    langs_detected = detect_and_group_by_language(cloudbusrt_signals_df)
    logger.info("Channels grouped by language")
