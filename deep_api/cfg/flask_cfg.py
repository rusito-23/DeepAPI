"""
Flask Extensions to load custom config.
Ref: https://gist.github.com/mattupstate/2046115
"""

from flask import Flask as BaseFlask, Config as BaseConfig


class Config(BaseConfig):
    """ Flask config enhanced - `from_cfg` method using the Main Config. """

    def from_cfg(self, cfg):
        self['ENV'] = cfg.FLASK_ENV
        self.__dict__.update(cfg.FLASK_CFG)
        for key in cfg.FLASK_CFG:
            self[key] = cfg.FLASK_CFG[key]


class Flask(BaseFlask):
    """ Extended version of `Flask` that implements custom config class """

    def make_config(self, instance_relative=False):
        root_path = self.root_path
        if instance_relative:
            root_path = self.instance_path
        return Config(root_path, self.default_config)
