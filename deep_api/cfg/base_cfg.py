"""
Base Configuration Node.
Updated the object dict based on the content of a YAML file.
"""

import os
import yaml
from utils.exception import ConfigError


class BaseCFG:
    """
    Base Configuration:
    Reads a YAML configuration from a given file.
    """

    def __init__(self, config_file):
        d = self._read_config(config_file)
        self.__dict__.update(d)

    def _read_config(self, config_file):
        """ Read the config as dict from file """
        with open(os.path.join(config_file), 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
                return cfg
            except yaml.YAMLError as exc:
                raise ConfigError(exc)
