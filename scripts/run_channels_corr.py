import logging
import pandas as pd
from clotho.extraction.loader import load_cloudburst_signals, COLUMNS_NAMES_SIGNALS
from clotho.pre_processing.create_dict import tuple_to_dict
from clotho.post_processing.clean_keys import clean_keys
from clotho.correlations.correlate_dicts import measure_similarity
from clotho.correlations.correlate_times import round_and_correlate
from clotho.post_processing.mix_weights import run_mix_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":

    ###########################################################################
    # Correlations ############################################################
    ###########################################################################

    # Load cloudburst signals #################################################
    logger.info("Loading cloudburst signals...")
    cloudburst_signals_df = load_cloudburst_signals()
    logger.info("Cloudburst signals loaded")
    ###########################################################################

    # Tuple to dict ###########################################################
    logger.info("Converting tuples to dict...")
    cloudburst_signals_df = tuple_to_dict(cloudburst_signals_df, COLUMNS_NAMES_SIGNALS)
    logger.info("Tuples converted to dict")
    ###########################################################################

    # Correlate signals time #################################################
    logger.info("Correlating signals time...")
    cloudburst_signals_corr = round_and_correlate(
        cloudburst_signals_df, "entity_id", "source_datetime"
    )
    logger.info("Signals time correlated")
    ###########################################################################

    ###########################################################################
    # Load channels with languages #############################################
    ###########################################################################

    # Load channels (json as dict)#############################################
    logger.info("Loading channels...")
    channels_lang = pd.read_csv("channels_lang.csv").to_dict("records")
    logger.info("Channels loaded")
    ###########################################################################

    # TODO ADD TIME_PUMP CORRELATION SPECIFICALLY #############################
    # SEE HOW TO DO IT WITH THE CORRELATE_TIMES FUNCTION

    ###########################################################################

    # Measure similarity on languages #########################################

    # Clean keys ##############################################################
    logger.info("Cleaning keys...")
    channels_lang_clean = clean_keys(channels_lang, "language", ["en", "Too short"])
    logger.info("Keys cleaned")
    ###########################################################################

    # Measure similarity on langs##############################################
    # TODO CALIBRATE SIMILARITY THE WEIGHTS ARE NOT GOOD YET
    logger.info("Measuring similarity...")
    channels_similarity = measure_similarity(
        channels_lang_clean, "language", "entity_id"
    )
    logger.info("Similarity measured")
    ###########################################################################

    # Mix weights for neo4j ###################################################
    logger.info("Running mix weights...")
    channels_mix_weights = run_mix_weights(
        [channels_similarity, cloudburst_signals_corr], "channel_corr.csv"
    )
    logger.info("Mix weights run")
    ###########################################################################
