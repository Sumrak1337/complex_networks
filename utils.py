import logging
import sys


def get_logger(name) -> logging.Logger:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.DEBUG)

    if not log.hasHandlers():
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)

    return log
