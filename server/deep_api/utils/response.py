"""
Base Response Class.
Used to normalize the JSON responses given by the app.
"""

from flask import jsonify


class Response:

    def __init__(self, result_image, success=True, message=None):
        self.content = {
            'success': success,
            'message': message,
            'result': result_image,
        }

    def json(self):
        return jsonify(self.content)
