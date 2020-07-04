"""
Flask App Routes.
"""

import io
from flask import Blueprint, request
from PIL import Image
from deep_dream.deep_dream import DeepDream
from utils.base64 import to_base64
from utils.response import Response


def create_blueprint(cfg):
    """ setup """

    bp = Blueprint('background', cfg.NAME)
    deep_dream = DeepDream(loi=cfg.MODEL.LOI,
                           weights_path=cfg.MODEL.WEIGHTS_PATH)

    """ routes """

    @bp.route('/deep/dream', methods=['POST'])
    def background_replacement():
        # read image
        image = request.files['image']
        image = Image.open(io.BytesIO(image.read()))

        # process
        result = deep_dream(image,
                            epochs=1,
                            learning_rate=1,
                            loi=['Mixed_5b', 'Mixed_6b'],
                            n_inceptions=2,
                            scale_factor=0.7,
                            blend_factor=0.3,
                            blur_radius=60)
        result = to_base64(result)

        # build response
        res = Response(result_image=result)
        return res.json()

    return bp
