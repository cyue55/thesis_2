import logging
import sys


def set_logger(log_file=None, distributed=False, rank=None, debug=False):
    """Set up the :mod:`logging` module.

    Parameters
    ----------
    log_file : str, optional
        Path to file to write logs to.
    distributed : bool, optional
        Whether the application is running in distributed mode.
    rank : int, optional
        Rank of the process in distributed mode.
    debug : bool, optional
        Whether to set the log level to :data:`logging.DEBUG`.

    """
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    prefix = "%(asctime)s [%(levelname)s:%(module)s]"
    if distributed:
        if rank is None:
            raise ValueError("must provide rank when distributed=True")
        f = _ContextFilter(rank)
        logger.addFilter(f)
        formatter = logging.Formatter(prefix + " [rank %(rank)s] %(message)s")
    else:
        formatter = logging.Formatter(prefix + " %(message)s")

    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class _ContextFilter(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True
