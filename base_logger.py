import logging

logger = None


def set_logger(new_logger, args):
    global logger
    logger = new_logger

    if "debug" in args:
        logger.setLevel(logging.DEBUG)
        print("Running on debug mode (debug arg)")
    else:
        logger.setLevel(logging.INFO)
        print("Running normally")


def get_logger():
    global logger
    return logger
