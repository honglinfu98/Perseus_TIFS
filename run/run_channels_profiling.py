import logging
from extraction_scripts.loader import load_cloudburst_signals
from profiling_scripts.process_channels_lenguages import detect_and_group_by_language

logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)

logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Loading cloudburst signals...")
    cloudbusrt_signals_df = load_cloudburst_signals()
    logger.info("Cloudburst signals loaded")
    logger.info("Detecting and grouping by channels language...")
    langs_detected = detect_and_group_by_language(cloudbusrt_signals_df)
    logger.info("Channels grouped by language")