import logging
import os
from logging import Logger
from pathlib import Path


def get_logger_default(out_path : str = str(Path(__file__).parent / "run.log") ) -> Logger:
    """
    Initialize the global logger with custom settings.
    """

    # Create a new logger
    logger = logging.getLogger("global_logger")
    logger.setLevel(logging.DEBUG)

    # Create the log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create log file if not exists
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    print(out_path)
    if not os.path.exists(out_path):
        os.system(f"touch {out_path}")
     
    fh = logging.FileHandler(out_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

