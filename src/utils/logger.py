import logging
from pathlib import Path
from datetime import datetime
import glob
import os

def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    file_prefix: str = "data_cleaning"  # you can set "data_ingestion", "feature_engineering" when calling
) -> logging.Logger:
    """
    Creates a logger with console and file handlers.
    Keeps only the last 10 log files.

    File name format: run_<timestamp>_<type>.log
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log file name: run_<timestamp>_<type>.log
    log_file = Path(log_dir) / f"run_{timestamp}_{file_prefix}.log"

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_level)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        # Clean old log files: keep only last 10 per type
        log_files = sorted(glob.glob(f"{log_dir}/run_*_{file_prefix}.log"), reverse=True)
        for old_file in log_files[10:]:
            os.remove(old_file)

    return logger
