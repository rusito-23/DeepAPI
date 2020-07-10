"""
Styles Configuration Node.
This configuration loads all base styles into an object.
"""

from cfg.base_cfg import BaseCFG
from utils.exception import UnknownStyle


class StyleCfg(BaseCFG):
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


class StylesCfg(BaseCFG):
    styles = {}

    def __init__(self, config_file):
        styles_dict = self._read_config(config_file)
        for k, v in styles_dict.items():
            self.styles[k] = StyleCfg(v)

    def __getitem__(self, style_name):
        if style_name not in self.styles:
            raise UnknownStyle(style_name)

        return self.styles[style_name]
