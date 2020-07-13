"""
Styles Routes
"""

import os
import io
import base64
from flask import Blueprint, jsonify
from PIL import Image


def im2base64(cfg, style_name):
    im_path = os.path.join(cfg.ASSETS_PATH, style_name + '.jpeg')
    image = Image.open(im_path).resize((32, 32))

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def create_blueprint(cfg):
    """ setup """

    bp = Blueprint('styles', cfg.NAME)

    """ routes """

    @bp.route('/deep/dream/styles', methods=['GET'])
    def get_styles():
        styles = [
            {
                'path': style_name,
                'name': cfg.STYLES_CFG[style_name].name,
                'description': cfg.STYLES_CFG[style_name].description,
                'icon': im2base64(cfg, style_name),
            }
            for style_name in cfg.STYLES_CFG.styles.keys()
            if style_name != 'base']

        return jsonify(styles)

    return bp
