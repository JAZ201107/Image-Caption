import logging


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logFormatter = logging.Formatter("%(asctime)s:[%(levelname)s]: %(message)s")
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler(log_path)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
