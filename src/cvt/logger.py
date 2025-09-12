import logging
from logging import getLogger

from rich.logging import RichHandler


def setup_logger(*, verbose: bool) -> None:
    root_logger = getLogger()
    root_logger.setLevel("DEBUG" if verbose else "INFO")
    root_logger.addHandler(RichHandler(omit_repeated_times=False, rich_tracebacks=True))
    logging.captureWarnings(capture=True)
