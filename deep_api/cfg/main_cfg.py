"""
Server Configuration Parsing.
"""

import os
import yaml
import argparse
import logging
from flask import Flask as BaseFlask, Config as BaseConfig
from utils.exception import ConfigError
from cfg.base_cfg import BaseCFG
from cfg.styles_cfg import StylesCfg


class CFG(BaseCFG):
    """ Main Configuration """

    def __init__(self, config_file='./support/config/config.yaml'):
        super(CFG, self).__init__(config_file)

        # setup flask
        self.FLASK_CFG = self._read_config(self.FLASK_CFG)[self.FLASK_ENV]

        # setup styles
        self.STYLES_CFG = StylesCfg(self.STYLES_CFG)

        # setup logging
        self.LOG_CFG = self._read_config(self.LOG_CFG)

        # create output folder
        if not os.path.isdir(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)

        # setup logging
        logging.config.dictConfig(self.LOG_CFG)
