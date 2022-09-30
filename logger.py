import logging

global logger
logger = logging.getLogger(__name__)


def setup_logger(args):
    global logger
    if "debug" in args:
        logger.setLevel(logging.DEBUG)
        print("Running on debug mode (debug arg)")
    else:
        logger.setLevel(logging.INFO)
        print("Running normally")

    return logger


def get_logger():
    global logger
    return logger
