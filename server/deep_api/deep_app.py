"""
Flask App.
"""

import logging
import logging.config
from config import CFG, Flask
from utils.exception import Error
from routes import (
    dream
)


def create_app():
    """ load config """

    cfg = CFG()

    """ create app """

    app = Flask(cfg.NAME)
    app.config.from_cfg(cfg)

    """ setup logging """

    logging.config.dictConfig(cfg.LOG)

    """ blueprints """

    app.register_blueprint(dream.create_blueprint(cfg))

    """ handlers """

    @app.errorhandler(Error)
    def error_handler(err):
        # custom errors
        app.logger.exception(err)
        return (f'Error: {err.message}', 500)

    @app.errorhandler(Exception)
    def exception_handler(exc):
        # other exceptions
        app.logger.exception(exc)
        return ('Unknown Error', 500)

    return app
