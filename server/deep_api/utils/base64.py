"""
Base 64 Converter
"""

import io
import base64


def to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    _bytes = buffered.getvalue()
    return base64.b64encode(_bytes).decode()
