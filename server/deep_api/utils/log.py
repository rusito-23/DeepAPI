"""
Logger Configuration.
"""

import os
import logging
import sys
import time


def config_logger(cfg):
    # format config
    logFormatter = logging.Formatter('%(asctime)s '
                                     '[%(levelname)s] '
                                     '%(message)s')

    # logger name
    rootLogger = logging.getLogger('ITT')

    # file config
    log_file = os.path.join(cfg.OUTPUT_FOLDER, f'{time.time()}.log')
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    # stdout config
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.getLevelName(cfg.LEVEL))
