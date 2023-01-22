"""
This script runs the flowork of the channels profiling.
"""
import logging
from clotho.extraction.loader import load_cloudburst_signals, COLUMNS_NAMES_SIGNALS
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.profiling.process_channels_lenguages import detect_lang_list_dict
from clotho.profiling.group_by_value_count import group_by_id_value_count
from clotho.correlations.dict_similarity import measure_similarity
from clotho.correlations.correlate_times import round_and_correlate
from clotho.post_processing.mix_weights import run_mix_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    # Download the signals data for channels profiling #######################
    logger.info("Loading cloudburst signals...")
    cloudburst_signals = load_cloudburst_signals()
    logger.info("Cloudburst signals loaded")

    logger.info("Number of signals extracted: %s", len(cloudburst_signals))

    logger.info("Converting to dictionary...")
    cloudburst_signals_df = tuple_to_dict(
        cloudburst_signals, keys=COLUMNS_NAMES_SIGNALS
    )
    logger.info("Cloudburst signals converted to dictionary")
    ###########################################################################

    # Correlate signals time #################################################
    logger.info("Correlating signals time...")
    cloudburst_signals_corr = round_and_correlate(
        cloudburst_signals_df, "entity_id", "source_datetime"
    )
    logger.info("Signals time correlated")
    ###########################################################################

    # Detect languages #########################################################
    logger.info("Detecting languages...")
    cloudburst_signals_featured = detect_lang_list_dict(cloudburst_signals_df)
    logger.info("Languages detected")

    logger.info("Grouping by entity_id and language...")
    channels_lang_count = group_by_id_value_count(
        cloudburst_signals_featured, "entity_id", "language"
    )
    logger.info("Grouped by entity_id and language")

    # Remove all the keys 'en' and 'too short' from the dictionary inside lenguages before measuring similarity
    def remove_keys(data):
        cleaned_data = []
        for d in data:
            if "language" in d:
                if "en" in d["language"]:
                    del d["language"]["en"]
                if "Too short" in d["language"]:
                    del d["language"]["Too short"]
                if d["language"]:
                    cleaned_data.append(d)
        return cleaned_data

    channels_lang_count_clean = remove_keys(channels_lang_count)

    logger.info("Measuring similarity...")
    channels_similarity = measure_similarity(channels_lang_count_clean)
    logger.info("Similarity measured")
    ###########################################################################

    # Mix weights for neo4j ###################################################
    logger.info("Running mix weights...")
    channels_mix_weights = run_mix_weights(
        [channels_similarity, cloudburst_signals_corr], "channel_corr.csv"
    )
    logger.info("Mix weights run")
    ###########################################################################
