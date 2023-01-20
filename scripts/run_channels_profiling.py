"""
This script runs the flowork of the channels profiling.
"""

import logging
from clotho.extraction.loader import load_cloudburst_signals, COLUMNS_NAMES_SIGNALS
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.profiling.process_channels_lenguages import detect_lang_list_dict
from clotho.profiling.group_by_value_count import group_by_id_value_count

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

if __name__ == "__main__":

    logger.info("Loading cloudburst signals...")
    cloudburst_signals = load_cloudburst_signals()
    logger.info("Cloudburst signals loaded")
    logger.info("Number of signals: %s", len(cloudburst_signals))

    logger.info("Converting to dictionary...")
    cloudburst_signals_df = tuple_to_dict(
        cloudburst_signals, keys=COLUMNS_NAMES_SIGNALS
    )
    logger.info("Cloudburst signals converted to dictionary")

    logger.info("Detecting languages...")
    cloudburst_signals_featured = detect_lang_list_dict(cloudburst_signals_df)
    logger.info("Languages detected")

    logger.info("Grouping by entity_id and language...")
    channels_leng_count = group_by_id_value_count(
        cloudburst_signals_featured, "entity_id", "language"
    )
    logger.info("Grouped by entity_id and language")
