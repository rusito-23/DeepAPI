"""
Logging extension to use Colors!
Taken from: https://stackoverflow.com/questions/384076
"""

import logging
from colorama import Back, Fore, Style

COLORS = {
    "WARNING": Fore.YELLOW,
    "INFO": Fore.CYAN,
    "DEBUG": Fore.BLUE,
    "CRITICAL": Fore.YELLOW,
    "ERROR": Fore.RED,
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, *, format, use_color):
        logging.Formatter.__init__(self, fmt=format)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        if self.use_color:
            levelname = record.levelname
            if hasattr(record, "color"):
                return f"{record.color}{msg}{Style.RESET_ALL}"
            if levelname in COLORS:
                return f"{COLORS[levelname]}{msg}{Style.RESET_ALL}"
        return msg
