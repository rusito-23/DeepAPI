"""
Flask App.
"""

import logging
import logging.config
from config import CFG, Flask
from utils.exception import Error
from utils.response import Response
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
        logger.error(err)
        return err.as_response().json()

    @app.errorhandler(Exception)
    def exception_handler(exc):
        # other exceptions
        logger.error(exc)
        return Response(result_image=None, success=False,
                        message='An error occurred from the server side.')\
            .json()

    return app
