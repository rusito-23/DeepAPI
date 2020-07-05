"""
Flask App Routes.
"""

import io
from flask import Blueprint, request, send_file
from PIL import Image
from algorithms.deep_dream import DeepDream


def create_blueprint(cfg):
    """ setup """

    bp = Blueprint('background', cfg.NAME)
    deep_dream = DeepDream(cfg=cfg.ALGORITHM)

    """ routes """

    @bp.route('/deep/dream', methods=['POST'])
    def background_replacement():
        # read image
        image = request.files['image']
        image = Image.open(io.BytesIO(image.read()))

        # dream!
        result = deep_dream(image)

        # build response
        result_bytes = io.BytesIO()
        result.save(result_bytes, 'JPEG', quality=80)
        result_bytes.seek(0)

        return send_file(result_bytes, mimetype='image/jpeg')

    return bp
