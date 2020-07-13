"""
Flask App.
"""

import logging
import logging.config
from cfg.flask_cfg import Flask
from cfg.main_cfg import CFG
from utils.exception import Error
from routes import (
    dream,
    styles
)


def create_app():
    """ load config """

    cfg = CFG()

    """ create app """

    app = Flask(cfg.NAME)
    app.config.from_cfg(cfg)

    """ blueprints """

    app.register_blueprint(dream.create_blueprint(cfg))
    app.register_blueprint(styles.create_blueprint(cfg))

    """ handlers """

    @app.errorhandler(Error)
    def error_handler(err):
        # custom errors
        app.logger.exception(err)
        return err.as_res()

    @app.errorhandler(Exception)
    def exception_handler(exc):
        # other exceptions
        app.logger.exception(exc)
        return ('Unknown Error', 500)

    return app
